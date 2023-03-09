import numpy as np
from ..dsp import ConverterFloatToQFormat

import logging
import logging.config
import yaml

logger = logging.getLogger(__name__)
# logger.debug = print

def vad(post_snr, prio_snr, qconverter: ConverterFloatToQFormat, threshold=0.15, snr_qformat:int=12):
    """
    Statistical Model-Based VAD

    vad_flag : int
        True (voice) or False (no voice)
    
    SNR Q format 12.20
    """
    if logger.level < 1:        
        post_snr_debug = qconverter.reconvert(post_snr)*(2**(snr_qformat-1))
        prio_snr_debug = qconverter.reconvert(prio_snr)*(2**(snr_qformat-1))
        div_debug = prio_snr_debug/(1+prio_snr_debug)
        multi_snr_debug = post_snr_debug*div_debug
        log_debug = np.log(1+prio_snr_debug)
        log_lambda_debug = (post_snr_debug*prio_snr_debug/(1+prio_snr_debug)) - np.log(1+prio_snr_debug)
    
    if post_snr.dtype not in (np.float32, np.float64, np.float16):
        if post_snr.dtype == np.int32: # Q 12.20
            one = 1 << (32-snr_qformat)
        elif post_snr.dtype == np.int16: # Q 12.4
            one = 1 << (16-snr_qformat)
        else:
            raise ValueError("The frame format is not supported in SNR Calcualtion...")

        if logger.level < 1:        
            log_prio_snr_log = np.zeros_like(post_snr)
            div_snr_log = np.zeros_like(post_snr)
            multi_snr_log = np.zeros_like(post_snr)

        log_lambda = np.zeros_like(post_snr)
        nfreq, nchannel = post_snr.shape
        for ch in range(nchannel):
            for nf in range(nfreq):
                _prio_snr = prio_snr[nf, ch]

                # print(f"\t Log Prio")
                # print(f"{qconverter.reconvert(np.array([_prio_snr+one, ], dtype=post_snr.dtype))*(2**(snr_qformat-1))}")
                # print(f"{prio_snr_debug[nf, ch]+1}")                

                prio_snr_log, shift_log = qconverter.log(one + _prio_snr, qformat=snr_qformat)

                # print(f"\t Log VAD")
                # print(f"{qconverter.reconvert(np.array([prio_snr_log, ], dtype=post_snr.dtype))*(2**shift_log)}")
                # print(f"{log_debug[nf, ch]}")                

                if shift_log >= snr_qformat:
                    raise ValueError("In VAD, SNR Calcualtion Q format overflowed....")
                prio_snr_log = prio_snr_log >> (snr_qformat-1-shift_log)
                
                ret, prio_snr_div, shift_div = qconverter.divide(_prio_snr, one+_prio_snr) # [TODO] shift is always 0?
                if shift_div >= snr_qformat:
                    raise ValueError("In VAD, SNR Calcualtion Q format overflowed....")

                if logger.level < 1:
                    log_prio_snr_log[nf, ch] = prio_snr_log
                    div_snr_log[nf, ch] = prio_snr_div
                    multi_snr_log[nf, ch] = qconverter.mult(post_snr[nf, ch], np.array(prio_snr_div, dtype=post_snr.dtype))

                log_lambda[nf, ch] = qconverter.mult(post_snr[nf, ch], np.array(prio_snr_div, dtype=post_snr.dtype)) - prio_snr_log

        logger.debug(f"\t Post SNR VAD")
        logger.debug(f"{qconverter.reconvert(post_snr.flatten()[:5])*(2**(snr_qformat-1))}")
        logger.debug(f"{post_snr_debug.flatten()[:5]}")

        logger.debug(f"\t Prio VAD")
        logger.debug(f"{qconverter.reconvert(prio_snr.flatten()[:5])*(2**(snr_qformat-1))}")
        logger.debug(f"{prio_snr_debug.flatten()[:5]}")

        logger.debug(f"\t Log VAD")
        logger.debug(f"{qconverter.reconvert(log_prio_snr_log.flatten()[:5])*(2**(snr_qformat-1))}")
        logger.debug(f"{log_debug.flatten()[:5]}")

        logger.debug(f"\t Div VAD")
        logger.debug(f"{qconverter.reconvert(div_snr_log.flatten()[:5])}")
        logger.debug(f"{div_debug.flatten()[:5]}")

        logger.debug(f"\t Multi VAD")
        logger.debug(f"{qconverter.reconvert(multi_snr_log.flatten()[:5])*(2**(snr_qformat-1))}")
        logger.debug(f"{multi_snr_debug.flatten()[:5]}")

        logger.debug(f"\t Log Lambda VAD")
        logger.debug(f"{qconverter.reconvert(log_lambda.flatten()[:5])*(2**(snr_qformat-1))}")
        logger.debug(f"{log_lambda_debug.flatten()[:5]}")
    else:
        log_lambda = (post_snr*prio_snr/(1+prio_snr)) - np.log(1+prio_snr)

    avg_lambda = np.average(log_lambda)
    if avg_lambda >= threshold:
        vad_flag = True
    else:
        vad_flag = False
    return vad_flag, avg_lambda

def snr_calc(input_fft_pwr, 
            noise_profile, 
            prev_post_snr, 
            prev_gain, 
            alpha, 
            update_alpha, 
            qconverter: ConverterFloatToQFormat, 
            snr_min,
            snr_qformat: int = 12,
            ):
    """
    Calculates A Posterioer SNR and A Priori SNR, Q format 12.20
    
    A Priori SNR estimation is based on Decision-Directed Approach

        Y(Noisy Signal) = X(Clean Signal) + D(Noise)
    
        E: Estmiation
        Priori SNR = E(|X(w_k)|^2)/E(|D(w_k)|^2)
        Posteriori SNR = |Y(w_k)|^2/E(|D(w_k)|^2)
        Instanctaneous SNR = |Y(w_k)|^2/E(|D(w_k)|^2) - 1
    """
    
    # print(input_fft_pwr[4:8, :], noise_profile)
    nfreq, nchannel = input_fft_pwr.shape
    post_snr = np.zeros_like(input_fft_pwr)
    inst_snr = np.zeros_like(input_fft_pwr)

    if post_snr.dtype not in (np.float32, np.float64, np.float16):
        one = np.ones(shape=(1,), dtype=post_snr.dtype)
        
        # Q1.31 -> Q12.20, Q1.15 -> Q12.4
        _alpha = alpha >> (snr_qformat-1)
        _update_alpha = update_alpha >> (snr_qformat-1)
        
        if post_snr.dtype == np.int32: # Q 12.20
            one = one << (32-snr_qformat)
        elif post_snr.dtype == np.int16: # Q 12.4
            one = one << (16-snr_qformat)
        else:
            raise ValueError("The frame format is not supported in SNR Calcualtion...")
        
        for ch in range(nchannel):
            for nf in range(nfreq):
                buf = 0
                ret, buf, shift_div = qconverter.divide(input_fft_pwr[nf, ch], noise_profile[nf, ch] + 1)
                
                if shift_div <= snr_qformat-1:
                    buf = buf >> (snr_qformat-1-shift_div)

                    # print("\t Noise Profile")
                    # if shift_div <= snr_qformat-1:
                    #     buf_debug = np.array([buf, ] , dtype=np.int32)
                    #     print(f"Shift {shift_div}, Div: {qconverter.reconvert(buf_debug)*(2**(snr_qformat-1))}", end=" ")
                    #     print(f"Div float {qconverter.reconvert(input_fft_pwr[nf, ch])/(qconverter.reconvert(noise_profile[nf, ch])+(2**-31))}")
                    #     print(f"Noise: {qconverter.reconvert(noise_profile[nf, ch])}, Signal: {qconverter.reconvert(input_fft_pwr[nf, ch])}")
                    # else:
                    #     raise ValueError(f"Over the SNR Range, Shift {shift_div}")
                else:
                    raise ValueError(f"SNR Q format is out of range, Shift {shift_div}")
                
                # snr  = noise / signal
                # post_snr = signal / noise signal
                # Instantaneous SNR, clean signal / noise signal
                post_snr[nf, ch] = buf       
                inst_snr[nf, ch] = buf - one # [TODO]

                # print("\t Noise Profile for post/inst snr")
                # if shift_div <= snr_qformat-1:
                #     buf_debug = np.array([post_snr[nf, ch], ] , dtype=np.int32)
                #     buf_float = qconverter.reconvert(input_fft_pwr[nf, ch])/(qconverter.reconvert(noise_profile[nf, ch])+(2**-31))

                #     print(f"post_snr: {qconverter.reconvert(buf_debug)*(2**(snr_qformat-1))}", end=" ")
                #     print(f"post_snr float {buf_float}")

                #     buf_debug = np.array([inst_snr[nf, ch], ] , dtype=np.int32)
                #     print(f"inst_snr: {qconverter.reconvert(buf_debug)*(2**(snr_qformat-1))}", end=" ")
                #     print(f"inst_snr float {buf_float-1}")
                

        if not isinstance(prev_gain, np.ndarray):
            # prio_snr = qconverter.mult(_update_alpha, inst_snr) << (snr_qformat-1)
            prio_snr = qconverter.mult(update_alpha, inst_snr)

            # post_snr_float = qconverter.reconvert(input_fft_pwr)/(qconverter.reconvert(noise_profile)+(2**-31))
            # inst_snr_float = post_snr_float - 1
            # update_alpha_float = qconverter.reconvert(update_alpha)
            # prio_snr_float = inst_snr_float * update_alpha_float

            # n=6
            # print(f"\t First. inst_snr")
            # print(f"{qconverter.reconvert(inst_snr.flatten()[:n])*(2**(snr_qformat-1))}")
            # print(f"{inst_snr_float.flatten()[:n]}")

            # print(f"\t First. update_alpha")
            # print(f"{qconverter.reconvert(_update_alpha.flatten()[:n])*(2**(snr_qformat-1))}")
            # print(f"{update_alpha_float.flatten()[:n]}")

            # print(f"\t First. Prio SNR")
            # print(f"{qconverter.reconvert(prio_snr.flatten()[:n])*(2**(snr_qformat-1))}")
            # print(f"{prio_snr_float.flatten()[:n]}")
        else:
            pow_gain = qconverter.mult(prev_gain, prev_gain)    
            # pow_gain_debug = pow_gain
            # prev_post_snr_power_gain = qconverter.mult(pow_gain, prev_post_snr)
            pow_gain = qconverter.mult(pow_gain, prev_post_snr)

            # alpha_pow_gain = qconverter.mult(alpha, pow_gain)
            # update_alpha_inst = qconverter.mult(update_alpha , inst_snr)

            prio_snr = qconverter.mult(alpha, pow_gain) + qconverter.mult(update_alpha , inst_snr)

            # print("\t Noise Profile")

            # post_snr_float = qconverter.reconvert(input_fft_pwr)/(qconverter.reconvert(noise_profile)+(2**-31))
            # inst_snr_float = post_snr_float - 1

            # pow_gain_float = qconverter.reconvert(prev_gain)
            # pow_gain_float = pow_gain_float*pow_gain_float

            # prev_post_snr_float = qconverter.reconvert(prev_post_snr)*(2**(snr_qformat-1))
            # prev_post_snr_power_gain_float = pow_gain_float*prev_post_snr_float
            # alpha_pow_gain_float = qconverter.reconvert(alpha)*prev_post_snr_power_gain_float
            # update_alpha_inst_snr_float = qconverter.reconvert(update_alpha)*inst_snr_float
            # prio_snr_float = alpha_pow_gain_float + update_alpha_inst_snr_float

            # n = 4
            # # print(f"  Power of gain")
            # # print(f"{qconverter.reconvert(pow_gain_debug.flatten()[:n])}")
            # # print(f"{pow_gain_float.flatten()[:n]}")

            # # print(f"\t prev_post_snr SNR")
            # # print(f"{qconverter.reconvert(prev_post_snr.flatten()[:n])*(2**(snr_qformat-1))}")
            # # print(f"{prev_post_snr_float.flatten()[:n]}")

            # print(f"\t Prev SNR * Pow Gain")
            # print(f"{qconverter.reconvert(prev_post_snr_power_gain.flatten()[:n])*(2**(snr_qformat-1))}")
            # print(f"{prev_post_snr_power_gain_float.flatten()[:n]}")

            # print(f" Alpha power of gain")
            # print(f"{qconverter.reconvert(alpha_pow_gain.flatten()[:n])*(2**(snr_qformat-1))}")
            # print(f"{alpha_pow_gain_float.flatten()[:n]}")

            # print(f" Updated alpha inst snr")
            # print(f"{qconverter.reconvert(update_alpha_inst.flatten()[:n])*(2**(snr_qformat-1))}")
            # print(f"{update_alpha_inst_snr_float.flatten()[:n]}")

            # print(f" Prio snr")
            # print(f"{qconverter.reconvert(prio_snr.flatten()[:n])*(2**(snr_qformat-1))}")
            # print(f"{prio_snr_float.flatten()[:n]}")
    else:
        # post_snr = input_fft_pwr/ (noise_profile + np.finfo(np.float32).eps) 
        post_snr = input_fft_pwr/ noise_profile
        inst_snr = post_snr - 1

        # [X] limitation on inst_snr to prevent distortion
        # inst_snr = np.maximum(inst_snr,0.1)
    
        # calculate A Priori SNR using Decision Directed approach
        prio_snr = alpha * (prev_gain**2 * prev_post_snr) + update_alpha * inst_snr

    # limitation on prio_snr to prevent distortion
    prio_snr = np.maximum(prio_snr, snr_min)
    return post_snr, prio_snr