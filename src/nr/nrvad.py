import numpy as np
from ..dsp import ConverterFloatToQFormat
from .nrprofile import(
    snr_calc,
    vad
)

import logging
import logging.config
import yaml
    
logger = logging.getLogger(__name__)
# logger.debug = print

class NRVAD(object):
    def __init__(self, config, qconverter: ConverterFloatToQFormat):        
        channel = config.setting.fft_channel
        nfreq_bin = config.params.window_size // 2 + 1
            
        self.qconverter = qconverter
        self.max_frame_count = config.nr.max_nframe_noise
        self.snr_qformat = config.nr.snr_qformat if config.nr.snr_qformat else 0
        
        snr_min = qconverter.convert(np.array([config.nr.snr_min, ] , dtype=np.float32))
        self.snr_min = snr_min >> (self.snr_qformat-1) if qconverter.qformat else snr_min
        vad_threshold = qconverter.convert(np.ones(shape=(1, ), dtype=np.float32) * config.nr.vad_threshold)
        self.vad_threshold = vad_threshold >> (self.snr_qformat-1) if qconverter.qformat else vad_threshold

        if qconverter.qformat == 8:
            dtype = np.int8
        elif qconverter.qformat == 15:
            dtype = np.int16
        elif qconverter.qformat == 31:
            dtype = np.int32
        else:
            dtype = np.float32

        self.alpha = qconverter.convert(np.ones(shape=(config.params.window_size//2+1, channel), dtype=np.float32)*0.98)
        self.update_alpha = qconverter.convert(np.ones(shape=(config.params.window_size//2+1, channel), dtype=np.float32)*0.02)

        # gain_min = np.ones(shape=(nfreq_bin, channel), dtype=np.float32)*config.nr.gain_min
        gain_min = np.ones(shape=(1, ), dtype=np.float32)*config.nr.gain_min
        self.gain_min = qconverter.convert(gain_min)

        self.prev_post_snr = qconverter.convert(np.ones(shape=(config.params.window_size//2+1, channel), dtype=np.float32))
        if config.params.qformat:
            self.prev_post_snr = self.prev_post_snr*2**(-config.nr.snr_qformat+1)

        self.prev_gain = 1 # [TODO] ?
        self.frame_count = 0

        self.noise_profile_sum = np.zeros(shape=(self.max_frame_count, nfreq_bin, channel), dtype=dtype)
        self.noise_profile = None
        self.profile = {"prio_snr":[], "post_snr":[], "avg_lambda":[]}
        
        self.vad_only = config.nr.vad_only
        self.nr_mode = None

        self.gain = np.zeros(shape=(config.params.window_size//2+1, channel), dtype=dtype)

    def forward(self, frame_fft):
        """
        Noise Reduction with VAD

        SNR : Q 12.20(SNR_Qformat in configuration)

        SNR/VAD SNR_Qformat -> SNR Ratio: Q 1.31 -> gain apply
        """
        # if all(frame_fft < 1e-6):
        #     return frame_fft

        qconverter = self.qconverter
        qformat = qconverter.qformat
        noise_profile = self.noise_profile
        frame_fft_pwr = qconverter.mag_squared(frame_fft)
        nfreq, channels = frame_fft_pwr.shape
        one = 1 << (qformat+1-self.snr_qformat) if self.snr_qformat else 1
        gain = self.gain

        logger.debug("\tMagnitude Squared")
        logger.debug(qconverter.reconvert(frame_fft_pwr.flatten()))
        logger.debug(f"Min: {np.min(qconverter.reconvert(frame_fft_pwr.flatten()))} Max: {np.max(qconverter.reconvert(frame_fft_pwr.flatten()))}")

        # Noise estimation
        if self.frame_count < self.max_frame_count:
            self.noise_profile_sum[self.frame_count] = frame_fft_pwr
            self.frame_count += 1        
            if self.frame_count == self.max_frame_count:
                self.noise_profile = np.mean(self.noise_profile_sum, axis=0, dtype=frame_fft.dtype)
                self.noise_profile_sum.fill(0)
            return frame_fft
        else:
            # print(noise_profile.shape) # currently noise profile at starting point is 0

            logger.debug("\t Noise Profile")
            logger.debug(qconverter.reconvert(noise_profile.flatten()))
            logger.debug(f"Min: {np.min(qconverter.reconvert(noise_profile.flatten()))} Max: {np.max(qconverter.reconvert(noise_profile.flatten()))}")

            # Use VAD to update noise profile
            post_snr, prio_snr = snr_calc(input_fft_pwr=frame_fft_pwr, 
                                          noise_profile=noise_profile,
                                          prev_post_snr=self.prev_post_snr,
                                          prev_gain=self.prev_gain,
                                          qconverter=qconverter,
                                          alpha=self.alpha,
                                          update_alpha=self.update_alpha,
                                          snr_min=self.snr_min,
                                          snr_qformat=self.snr_qformat,
                                          )
        
            vad_flag, avg_lambda = vad(post_snr=post_snr, 
                                    prio_snr=prio_snr, 
                                    qconverter=qconverter, 
                                    threshold=self.vad_threshold,
                                    snr_qformat=self.snr_qformat,
                                    )
            
            # print(vad_flag)

            if not vad_flag:
                # Speech Enhancement Theory and Practice 582 page
                # D_k(i) - (1-\beta)*Y_k^2(i) + \beta D_k(i-1)
                # \beta = 0.98
                # D_k(i) is the noise power specturm in frame i (for frequency bin k)
                # Y^2_k(i)  is the noisy speech power sepctrum
                self.noise_profile = qconverter.mult(self.update_alpha, frame_fft_pwr) + qconverter.mult(self.alpha, noise_profile)
            
            # gain calculation
            if qformat in (7, 15, 31):
                for ch in range(channels):
                    for nf in range(nfreq):
                        _prio_snr = prio_snr[nf, ch]
                        _, _gain, shift_div = qconverter.divide(_prio_snr, one + _prio_snr) 
                        if shift_div > 1:
                            raise ValueError("Gain is incremental Gain, it should be constraint...")

                        # limit gain to a minimum of -16dB (0.158)
                        gain[nf, ch] = np.maximum(_gain, self.gain_min)
            else:
                gain = prio_snr / (1+prio_snr)
                gain = np.maximum(gain, self.gain_min)

            # print(gain[4:8, :], self.gain_min[4:8, :])
            
            self.profile["prio_snr"].append(prio_snr)
            self.profile["post_snr"].append(post_snr)
            self.profile["avg_lambda"].append(avg_lambda)

            # Store gain and post-snr
            self.prev_gain = gain
            self.prev_post_snr = post_snr

            if qformat in (15, 31):
                logger.debug("\tPost")
                logger.debug(qconverter.reconvert(post_snr.flatten())*2**(self.snr_qformat-1))
                logger.debug("\tPrio")
                logger.debug(qconverter.reconvert(prio_snr.flatten())*2**(self.snr_qformat-1))
                logger.debug("\t VAD")
                logger.debug(f"Flag {vad_flag}, {avg_lambda.flatten()}")
                logger.debug("\t Gain")
                logger.debug(f"{qconverter.reconvert(gain.flatten())}")
            else:
                logger.debug("\tPost")
                logger.debug(qconverter.reconvert(post_snr.flatten()))
                logger.debug("\tPrio")
                logger.debug(qconverter.reconvert(prio_snr.flatten()))
                logger.debug("\t VAD")
                logger.debug(f"Flag {vad_flag}, {avg_lambda.flatten()}")
                logger.debug("\t Gain")
                logger.debug(f"{qconverter.reconvert(gain.flatten())}")
                logger.debug("\tPost MAX")
                logger.debug(np.max(qconverter.reconvert(post_snr.flatten())))
                logger.debug("\tPrio MAX")
                logger.debug(np.max(qconverter.reconvert(prio_snr.flatten())))

            if self.vad_only:
                if vad_flag:
                    return frame_fft # np.ones_like(frame_fft)*0.95
                else:
                    return np.zeros_like(frame_fft)
            else:
                # apply gain 
                frame_fft[0::2, ...] = qconverter.mult(frame_fft[0::2, ...], gain)
                frame_fft[1::2, ...] = qconverter.mult(frame_fft[1::2, ...], gain)
                return frame_fft 
