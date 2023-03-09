import numpy as np
from ..dsp import ConverterFloatToQFormat

class NRSpectralGate(object):
    def __init__(self, config, qconverter: ConverterFloatToQFormat):
        channel = config.setting.fft_channel
        nfreq_bin = config.params.window_size // 2 + 1
        self.qconverter = qconverter

        if qconverter.qformat == 8:
            dtype = np.int8
        elif qconverter.qformat == 15:
            dtype = np.int16
        elif qconverter.qformat == 31:
            dtype = np.int32
        else:
            dtype = np.float32

        self.stft = np.zeros(shape=(config.nr.max_nframe_statistic, nfreq_bin, channel), dtype=dtype)
        self.frame_count = 0
        self.max_frame_count = config.nr.max_nframe_statistic
        
        std_threshold = qconverter.convert(np.array(config.nr.std_th, dtype=np.float32)) # Q2.X
        self.std_threshold = std_threshold >> 1 if qconverter.qformat else std_threshold
        
        self.alpha = qconverter.convert(np.array(config.nr.gain, dtype=np.float32))
        self.updated_alpha = qconverter.convert(np.array(1-config.nr.gain, dtype=np.float32))
        

    def forward(self, frame_fft):

        qconverter = self.qconverter
        qformat = qconverter.qformat
        frame_fft_pwr = qconverter.mag(frame_fft) # TODO Energy
        nfreq, channels = frame_fft_pwr.shape

        cnt_frame = self.frame_count

        # Statistical analysis
        # X fixedpoint imeplmentation, performance is not good 
        if cnt_frame < self.max_frame_count:
            self.stft[cnt_frame, ...] = frame_fft_pwr
            self.frame_count += 1
            return frame_fft
        
        stft_pwr = self.stft    
        if qformat:
            raise NotImplementedError
        else:
            mean_pwr = np.mean(stft_pwr, axis=0)
            std_pwr = np.std(stft_pwr, axis=0)
        
        # Apply gain
        noise_thresh_pwr = mean_pwr + std_pwr * self.std_threshold
        frame_mask = frame_fft_pwr > noise_thresh_pwr
        frame_mask = frame_mask * self.updated_alpha + np.ones_like(frame_fft_pwr) * self.alpha
        frame_fft[0::2, ...] = qconverter.mult(frame_fft[0::2, ...], frame_mask)
        frame_fft[1::2, ...] = qconverter.mult(frame_fft[1::2, ...], frame_mask)

        # print(f"\t {stft_mask[cnt_frame, ...].flatten()[:5]}")
        # print(f"\t {cnt_frame}, {self.max_frame_count}")
        # print(f"\t {mean_pwr.flatten()[:5]}")
        # print(f"\t {noise_thresh_pwr.flatten()[:5]}")
        # print(f"\t {stft_pwr[cnt_frame, ...].flatten()[:5]}")
        # print("-"*30)
        # print(f"\t {stft_mask[cnt_frame, ...].flatten()[:5]}")
        
                                               
        return frame_fft