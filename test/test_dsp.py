import cmsisdsp as dsp 
import numpy as np
import unittest
import warnings
from numpy.testing import assert_allclose

class NumericalSanityCheck(unittest.TestCase):
    def test_basic_recip_q15(self):
        """
        python -m unittest -v test.test_dsp.NumericalSanityCheck.test_basic_recip_q15
        """
        print()

        recipQ15=np.array([0x7F03, 0x7D13, 0x7B31, 0x795E, 0x7798, 0x75E0,
        0x7434, 0x7294, 0x70FF, 0x6F76, 0x6DF6, 0x6C82,
        0x6B16, 0x69B5, 0x685C, 0x670C, 0x65C4, 0x6484,
        0x634C, 0x621C, 0x60F3, 0x5FD0, 0x5EB5, 0x5DA0,
        0x5C91, 0x5B88, 0x5A85, 0x5988, 0x5890, 0x579E,
        0x56B0, 0x55C8, 0x54E4, 0x5405, 0x532B, 0x5255,
        0x5183, 0x50B6, 0x4FEC, 0x4F26, 0x4E64, 0x4DA6,
        0x4CEC, 0x4C34, 0x4B81, 0x4AD0, 0x4A23, 0x4978,
        0x48D1, 0x482D, 0x478C, 0x46ED, 0x4651, 0x45B8,
        0x4521, 0x448D, 0x43FC, 0x436C, 0x42DF, 0x4255,
        0x41CC, 0x4146, 0x40C2, 0x4040])

        s, v=dsp.arm_recip_q15(int(0x2000),recipQ15)
        # s: 
        # v: 
        print("1 / 0.25")
        print(s)
        print("%04X -> %f" % (v,((v<<s)/(1<<15))))
        print("----\n")

        # s,v=dsp.arm_recip_q15(1,recipQ15)
        # print("1 / (1 << 15)")
        # print(s)
        # print("%04X -> %f" % (v,((v<<s)/(1<<15))))
        # print(1<<15)
        # print("----\n")
    
        print(recipQ15)
        print([1<<i for i in range(16)])

    def test_cmsis_dsp_fft(self):
        """
        python -m unittest -v test.test_dsp.NumericalSanityCheck.test_cmsis_dsp_fft

        CMSIS tested with error rate(testrfft_all.py)
        
            Q15:  rtol=1e-6, atol=1e-2  
            Q31:  rtol=1e-6, atol=1e-6

        fft Wrapper          : cmsisdsp_transform.c
        fft instance(struct) : transform_functions.h
        fft init             : arm_rfft_init_q15.c  arm_rfft_init_q31.c
        fft/ifft function    : arm_rfft_q15.c       arm_rfft_q31.c
        fft cofficient table : arm_common_tables.c
        """
        import librosa
        import resampy 
        import matplotlib.pyplot as plt
        print()
        wav, sr = librosa.load(librosa.ex("trumpet"))
        wav = resampy.resample(wav, sr, 16000)
        sr = 16000
        window_size = 256 # 128, 256, 512, 1024
        wav = wav.flatten()
        wav = wav[:window_size]

        # cosine wav
        # wav = np.cos(2 * np.pi * np.arange(window_size) / window_size)*np.cos(0.2*2 * np.pi * np.arange(window_size) / window_size)
        
        error_rate_q15 = 1e-3
        error_rate_q31 = 1e-5
        error_rate_q7 = 1e-1

        """Numpy FFT, iFFT 
        """
        # fft float ~ numpy
        # rfft, cfft
        wav_rfft = np.fft.rfft(a=wav, n=window_size)
        wav_cfft = np.fft.fft(a=wav, n=window_size)
        
        wav_rfft_bin = np.abs(wav_rfft)
        wav_cfft_bin = np.abs(wav_cfft)

        wav_irfft = np.fft.irfft(a=wav_rfft, n=window_size)
        wav_icfft = np.fft.ifft(a=wav_cfft, n=window_size)

        assert_allclose(wav_irfft, wav, atol=error_rate_q31)
        assert_allclose(wav_icfft, wav, atol=error_rate_q31)
        
        # plt.plot(wav)
        # plt.plot(wav_irfft)
        # plt.plot(wav_icfft)
        # plt.show()
        
        print("Pass wav reconstruction using Numpy FFT and iFFT ")

        """Wavform --------> Fixed point Wavform
        """
        # wav
        wav_q15 = dsp.arm_float_to_q15(wav)
        wav_q31 = dsp.arm_float_to_q31(wav)
        wav_q7 = dsp.arm_float_to_q7(wav)
        
        wav_q15_revert = dsp.arm_q15_to_float(wav_q15)
        wav_q31_revert = dsp.arm_q31_to_float(wav_q31)
        wav_q7_revert = dsp.arm_q7_to_float(wav_q7)
        
        assert_allclose(wav_q7_revert, wav, atol=error_rate_q7)
        assert_allclose(wav_q15_revert, wav, atol=error_rate_q15)
        assert_allclose(wav_q31_revert, wav, atol=error_rate_q31)

        # plt.plot(wav_q15_revert)
        # plt.plot(wav_q31_revert)        
        # plt.show()

        print("Pass floating point to fixed point")

        """Convolve
        """
        wav_conv = np.convolve(wav, wav, mode="full")
        wav_conv_q15 = dsp.arm_conv_q15(wav_q15, len(wav_q15), wav_q15, len(wav_q15))
        wav_conv_q31 = dsp.arm_conv_q31(wav_q31, len(wav_q31), wav_q31, len(wav_q31))
        wav_conv_q7  = dsp.arm_conv_q7(wav_q7, len(wav_q7), wav_q7, len(wav_q7))

        wav_conv_q15_revert = dsp.arm_q15_to_float(wav_conv_q15)
        wav_conv_q31_revert = dsp.arm_q31_to_float(wav_conv_q31)
        wav_conv_q7_revert = dsp.arm_q7_to_float(wav_conv_q7)
        
        assert_allclose(wav_conv_q15_revert, wav_conv, atol=error_rate_q15)
        assert_allclose(wav_conv_q31_revert, wav_conv, atol=error_rate_q31)
        assert_allclose(wav_conv_q7_revert, wav_conv, atol=error_rate_q7)

        print("Pass Convolve")

        """Wavform --------> FFT
        """
        # fft real and imag
        # rfft
        rfft_instance_q15 = dsp.arm_rfft_instance_q15()
        status = dsp.arm_rfft_init_q15(rfft_instance_q15, window_size, 0, 1)
        rfft_instance_q31 = dsp.arm_rfft_instance_q31()
        status = dsp.arm_rfft_init_q31(rfft_instance_q31, window_size, 0, 1)

        rfft_1_q15 = dsp.arm_rfft_q15(rfft_instance_q15, wav_q15) 
        rfft_1_q31 = dsp.arm_rfft_q31(rfft_instance_q31, wav_q31)
        
        rfft_1_q15_revert = dsp.arm_q15_to_float(rfft_1_q15)*window_size
        rfft_1_q31_revert = dsp.arm_q31_to_float(rfft_1_q31)*window_size


        wav_rfft_reference = np.zeros(shape=(window_size*2+2, ), dtype=np.float32)
        wav_rfft_reference[:window_size+2][0::2] = wav_rfft.real
        wav_rfft_reference[window_size+2:][0::2] = np.flip(wav_rfft[:-1].real)
        wav_rfft_reference[:window_size+2][1::2] = wav_rfft.imag
        wav_rfft_reference[window_size+2:][1::2] = -np.flip(wav_rfft[:-1].imag)

        print("\t FFT")
        if wav_rfft_reference.shape != rfft_1_q15.shape: 
            warnings.warn(f"CMSIS rfft size is different...{wav_rfft_reference.shape} vs CMSIS {rfft_1_q15_revert.shape}")
        print("Error Max in fft Q15: ", np.max(np.abs(wav_rfft_reference[:window_size+2]-rfft_1_q15_revert[:window_size+2])))
        print("Error Max in fft Q31: ", np.max(np.abs(wav_rfft_reference[:window_size+2]-rfft_1_q31_revert[:window_size+2])))
        
        # # [TODO] cfft: segmentation fault
        # cfft_instance_q15 = dsp.arm_cfft_instance_q15()
        # status = dsp.arm_cfft_init_q15(cfft_instance_q15, window_size)
        
        # cfft_instance_q31 = dsp.arm_cfft_instance_q31()
        # status = dsp.arm_cfft_init_q31(cfft_instance_q31, window_size)
        
        # cfft_1_q15 = dsp.arm_cfft_q15(cfft_instance_q15, wav_q15, False, True) 
        # cfft_1_q31 = dsp.arm_cfft_q31(cfft_instance_q31, wav_q31, False, True)
        
        # cfft_1_q15_revert = dsp.arm_q15_to_float(cfft_1_q15)*window_size
        # cfft_1_q31_revert = dsp.arm_q31_to_float(cfft_1_q31)*window_size

        """FFT --------> Amplitude
        """
        wav_rfft_reference_q15 = dsp.arm_float_to_q15(wav_rfft_reference/window_size)
        wav_rfft_reference_q31 = dsp.arm_float_to_q31(wav_rfft_reference/window_size)

        # float fft real and imag ~ amplitude
        fft_bins_1_q15 = dsp.arm_cmplx_mag_q15(wav_rfft_reference_q15)[:window_size // 2 + 1]
        fft_bins_1_q31 = dsp.arm_cmplx_mag_q31(wav_rfft_reference_q31)[:window_size // 2 + 1]

        fft_bins_1_q15_revert = dsp.arm_q15_to_float(fft_bins_1_q15)*window_size*2
        fft_bins_1_q31_revert = dsp.arm_q31_to_float(fft_bins_1_q31)*window_size*2

        print("\t FFT Magnitude float -> fixed")
        print("Error Max in abs float fft -> cmsis abs Q15: ", np.max(np.abs(fft_bins_1_q15_revert-wav_rfft_bin)))
        print("Error Max in abs float fft -> cmsis abs Q31: ", np.max(np.abs(fft_bins_1_q31_revert-wav_rfft_bin)))
        
        # fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
        # ax0.plot(wav_rfft_bin)
        # ax1.plot(fft_bins_1_q15_revert)
        # ax2.plot(fft_bins_1_q31_revert)
        
        # fft real and imag ~ amplitude
        fft_bins_1_q15 = dsp.arm_cmplx_mag_q15(rfft_1_q15)[:window_size // 2 + 1]
        fft_bins_1_q31 = dsp.arm_cmplx_mag_q31(rfft_1_q31)[:window_size // 2 + 1]

        fft_bins_1_q15_revert = dsp.arm_q15_to_float(fft_bins_1_q15)*window_size*2
        fft_bins_1_q31_revert = dsp.arm_q31_to_float(fft_bins_1_q31)*window_size*2

        print("\t FFT Magnitude fixed -> fixed")
        print("Error Max in cmsis fft -> cmsis abs fft Q15: ", np.max(np.abs(fft_bins_1_q15_revert-wav_rfft_bin)))
        print("Error Max in cmsis fft -> cmsis abs fft Q31: ", np.max(np.abs(fft_bins_1_q31_revert-wav_rfft_bin)))
        
        """Ampltude --------> Noramlized Amplitude
        """
        fft_bins_1_q15_norm = fft_bins_1_q15 >> int(np.log2(window_size//2))
        fft_bins_1_q31_norm = fft_bins_1_q31 >> int(np.log2(window_size//2))
        wav_rfft_bin_norm = wav_rfft_bin / (window_size//2)

        fft_bins_1_q15_norm_revert = dsp.arm_q15_to_float(fft_bins_1_q15_norm)*window_size*2
        fft_bins_1_q31_norm_revert = dsp.arm_q31_to_float(fft_bins_1_q31_norm)*window_size*2

        print("\t Fixed FFT Normalized Magnitude")
        print("Error Max in cmsis fft -> cmsis abs fft Q15 -> normalized: ", np.max(np.abs(fft_bins_1_q15_norm_revert-wav_rfft_bin_norm)))
        print("Error Max in cmsis fft -> cmsis abs fft Q31 -> normalized: ", np.max(np.abs(fft_bins_1_q31_norm_revert-wav_rfft_bin_norm)))
        

        """FFT --------> iFFT
        """
        # float fft real and imag ~ ifft
        # rfft
        irfft_instance_q15 = dsp.arm_rfft_instance_q15()
        status = dsp.arm_rfft_init_q15(irfft_instance_q15, window_size, 1, 1)

        irfft_instance_q31 = dsp.arm_rfft_instance_q31()
        status = dsp.arm_rfft_init_q31(irfft_instance_q31, window_size, 1, 1)
        
        # plt.plot(dsp.arm_q15_to_float(rfft_1_q15_full)*window_size)
        # plt.plot(wav_rfft_reference)

        irfft_1_q15 = dsp.arm_rfft_q15(irfft_instance_q15, wav_rfft_reference_q15) 
        irfft_1_q31 = dsp.arm_rfft_q31(irfft_instance_q31, wav_rfft_reference_q31)
        
        irfft_1_q15_revert = dsp.arm_q15_to_float(irfft_1_q15)*window_size
        irfft_1_q31_revert = dsp.arm_q31_to_float(irfft_1_q31)*window_size
        

        print("\t iFFT float -> fixed")
        if irfft_1_q15_revert.shape != wav.shape:
            warnings.warn(f"Q15, Q31 inverse fourier transform length is shorter than original")
            print("Error Max in float fft -> ifft Q15: ", np.max(np.abs(wav[...,:irfft_1_q15_revert.shape[-1]]-irfft_1_q15_revert)))
            print("Error Max in float fft -> ifft Q31: ", np.max(np.abs(wav[...,:irfft_1_q15_revert.shape[-1]]-irfft_1_q31_revert)))
        else:
            print("Error Max in float fft -> ifft Q15: ", np.max(np.abs(wav-irfft_1_q15_revert)))
            print("Error Max in float fft -> ifft Q31: ", np.max(np.abs(wav-irfft_1_q31_revert)))
            
        # fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
        # ax0.plot(wav)
        # ax1.plot(irfft_1_q15_revert)
        # ax2.plot(irfft_1_q31_revert)
        
        # fft cmsis_dsp real and imag ~ ifft
        # rfft
        rfft_instance_q15 = dsp.arm_rfft_instance_q15()
        status = dsp.arm_rfft_init_q15(rfft_instance_q15, window_size, 1, 1)
        rfft_instance_q31 = dsp.arm_rfft_instance_q31()
        status = dsp.arm_rfft_init_q31(rfft_instance_q31, window_size, 1, 1)

        irfft_1_q15 = dsp.arm_rfft_q15(rfft_instance_q15, rfft_1_q15) 
        irfft_1_q31 = dsp.arm_rfft_q31(rfft_instance_q31, rfft_1_q31)
        
        irfft_1_q15_revert = dsp.arm_q15_to_float(irfft_1_q15)*window_size
        irfft_1_q31_revert = dsp.arm_q31_to_float(irfft_1_q31)*window_size

        print("\t iFFT fixed -> fixed")
        if irfft_1_q15_revert.shape != wav.shape:
            warnings.warn(f"Q15, Q31 inverse fourier transform length is shorter than original")
            print("Error Max in fft Q15 -> ifft Q15: ", np.max(np.abs(wav[...,:irfft_1_q15_revert.shape[-1]]-irfft_1_q15_revert)))
            print("Error Max in fft Q31 -> ifft Q31: ", np.max(np.abs(wav[...,:irfft_1_q15_revert.shape[-1]]-irfft_1_q31_revert)))
        else:
            print("Error Max in fft Q15 -> ifft Q15: ", np.max(np.abs(wav-irfft_1_q15_revert)))
            print("Error Max in fft Q31 -> ifft Q31: ", np.max(np.abs(wav-irfft_1_q31_revert)))


        # fft cmsis_dsp real and imag ~ ifft using f bin until nyquist frequnecy and other's filled with flip
        # rfft
        rfft_instance_q15 = dsp.arm_rfft_instance_q15()
        status = dsp.arm_rfft_init_q15(rfft_instance_q15, window_size, 1, 1)
        rfft_instance_q31 = dsp.arm_rfft_instance_q31()
        status = dsp.arm_rfft_init_q31(rfft_instance_q31, window_size, 1, 1)

        irfft_1_q15 = dsp.arm_rfft_q15(rfft_instance_q15, rfft_1_q15[:window_size+2]) 
        irfft_1_q31 = dsp.arm_rfft_q31(rfft_instance_q31, rfft_1_q31[:window_size+2])
        
        irfft_1_q15_revert = dsp.arm_q15_to_float(irfft_1_q15)*window_size
        irfft_1_q31_revert = dsp.arm_q31_to_float(irfft_1_q31)*window_size

        print("\t iFFT Half flip fixed -> fixed")
        if irfft_1_q15_revert.shape != wav.shape:
            warnings.warn(f"Q15, Q31 inverse fourier transform length is shorter than original")
            print("Error Max in fft Q15 -> ifft Q15: ", np.max(np.abs(wav[...,:irfft_1_q15_revert.shape[-1]]-irfft_1_q15_revert)))
            print("Error Max in fft Q31 -> ifft Q31: ", np.max(np.abs(wav[...,:irfft_1_q15_revert.shape[-1]]-irfft_1_q31_revert)))
        else:
            print("Error Max in fft Q15 -> ifft Q15: ", np.max(np.abs(wav-irfft_1_q15_revert)))
            print("Error Max in fft Q31 -> ifft Q31: ", np.max(np.abs(wav-irfft_1_q31_revert)))

    def test_dsp_operation(self):
        """
        python -m unittest -v test.test_dsp.NumericalSanityCheck.test_dsp_operation
        """
        from src.dsp import ConverterFloatToQFormat
        import librosa 
        import resampy 
        print()
        wav, sr = librosa.load(librosa.ex("trumpet"))
        wav = resampy.resample(wav, sr, 16000)
        sr = 16000
        window_size = 256
        wav = wav.flatten()
        wav = np.expand_dims(wav, axis=-1)
        wav = wav[..., :window_size, :]

        print("Input shape: ", wav.shape)
        # Init
        converter_q15 = ConverterFloatToQFormat(window_size=window_size,
                                            qformat=15,
                                            win_type="hann")
        converter_q31 = ConverterFloatToQFormat(window_size=window_size,
                                            qformat=31,
                                            win_type="hann")
        converter_q7 = ConverterFloatToQFormat(window_size=window_size,
                                            qformat=7,
                                            win_type="hann")
        converter_float = ConverterFloatToQFormat(window_size=window_size,
                                            qformat=0,
                                            win_type="hann")

        error_rate_q15 = 1e-3 # max error threshold 1e-3
        error_rate_q31 = 1e-5 # max error threshold 1e-5
        error_rate_q7 = 1e-1

        # Conversion
        wav_q15 = converter_q15.convert(wav)
        wav_q31 = converter_q31.convert(wav)
        wav_q7 = converter_q7.convert(wav)
        wav_float = converter_float.convert(wav)

        assert (wav_float-converter_q15.reconvert(wav_q15) < error_rate_q15).all()
        assert (wav_float-converter_q31.reconvert(wav_q31) < error_rate_q31).all()
        assert (wav_float-converter_q7.reconvert(wav_q7) < error_rate_q7).all()

        print("Conversion Pass !")
        
        # Add
        wav_q15_add = converter_q15.add(wav_q15, wav_q15)
        wav_q31_add = converter_q31.add(wav_q31, wav_q31)
        wav_q7_add = converter_q7.add(wav_q7, wav_q7)
        wav_float_add  = converter_float.add(wav_float, wav_float)
        assert (wav_float_add-converter_q15.reconvert(wav_q15_add) < error_rate_q15).all()
        assert (wav_float_add-converter_q31.reconvert(wav_q31_add) < error_rate_q31).all()
        assert (wav_float_add-converter_q7.reconvert(wav_q7_add) < error_rate_q7).all()

        print("Add Pass !")

        # Subtract
        wav_q15_sub = converter_q15.sub(wav_q15, wav_q15)
        wav_q31_sub = converter_q31.sub(wav_q31, wav_q31)
        wav_q7_sub = converter_q7.sub(wav_q7, wav_q7)
        wav_float_sub  = converter_float.sub(wav_float, wav_float)

        assert (wav_float_sub-converter_q15.reconvert(wav_q15_sub) < error_rate_q15).all()
        assert (wav_float_sub-converter_q31.reconvert(wav_q31_sub) < error_rate_q31).all()
        assert (wav_float_sub-converter_q7.reconvert(wav_q7_sub) < error_rate_q7).all()

        print("Subtract Pass !")

        # Multiply
        wav_q15_mult = converter_q15.mult(wav_q15, wav_q15)
        wav_q31_mult = converter_q31.mult(wav_q31, wav_q31)
        wav_q7_mult = converter_q7.mult(wav_q7, wav_q7)
        wav_float_mult  = converter_float.mult(wav_float, wav_float)
        
        assert (wav_float_mult-converter_q15.reconvert(wav_q15_mult) < error_rate_q15).all()
        assert (wav_float_mult-converter_q31.reconvert(wav_q31_mult) < error_rate_q31).all()
        assert (wav_float_mult-converter_q7.reconvert(wav_q7_mult) < error_rate_q7).all()

        print("Multiply Pass !")

        # Divide
        ret_q15_divide, wav_q15_divide, shift_q15_divide = converter_q15.divide(wav_q15[2, 0], wav_q15[2, 0])
        ret_q31_divide, wav_q31_divide, shift_q31_divide = converter_q31.divide(wav_q31[2, 0], wav_q31[2, 0])
        # wav_q7_divide = converter_q7.divide(wav_q7[2, 0], wav_q7[2, 0])
        ret_float_divide, wav_float_divide, shift_float_divide  = converter_float.divide(wav_float[2, 0], wav_float[2, 0])

        wav_q15_divide_reconvert = (converter_q15.reconvert(np.array([wav_q15_divide, ], dtype=wav_q15.dtype))*2**shift_q15_divide)[0]
        wav_q31_divide_reconvert = (converter_q31.reconvert(np.array([wav_q31_divide, ], dtype=wav_q31.dtype))*2**shift_q31_divide)[0]

        assert (wav_float_divide-wav_q15_divide_reconvert < error_rate_q15).all()
        assert (wav_float_divide-wav_q31_divide_reconvert < error_rate_q31).all()

        print("Divide Pass !")

        # Sqrt
        ret_q15_float, wav_q15_sqrt = converter_q15.sqrt(wav_q15[2, 0])
        ret_q15_float, wav_q31_sqrt = converter_q31.sqrt(wav_q31[2, 0])
        # ret_q15_float, wav_q7_sqrt = converter_q7.sqrt(wav_q7[2, 0])
        ret_q15_float, wav_float_sqrt  = converter_float.sqrt(wav_float[2, 0])

        wav_q15_sqrt_reconvert = (converter_q15.reconvert(np.array([wav_q15_sqrt, ], dtype=wav_q15.dtype)))[0]
        wav_q31_sqrt_reconvert = (converter_q31.reconvert(np.array([wav_q31_sqrt, ], dtype=wav_q31.dtype)))[0]
        
        assert (wav_float_sqrt- wav_q15_sqrt_reconvert< error_rate_q15).all()
        assert (wav_float_sqrt- wav_q31_sqrt_reconvert < error_rate_q31).all()

        print("Sqrt Pass !")

        # Shift
        shift = -4 # [TODO] overflow, underflow flag
        wav_q15_shift = converter_q15.shift(wav_q15, shift)
        wav_q31_shift = converter_q31.shift(wav_q31, shift)
        wav_q7_shift = converter_q7.shift(wav_q7, shift)
        wav_float_shift  = converter_float.shift(wav_float, shift)

        wav_q15_shift_reconvert = converter_q15.reconvert(wav_q15_shift)
        wav_q31_shift_reconvert = converter_q31.reconvert(wav_q31_shift)
        wav_q7_shift_reconvert = converter_q7.reconvert(wav_q7_shift)

        assert (wav_float_shift- wav_q15_shift_reconvert< error_rate_q15).all()
        assert (wav_float_shift- wav_q31_shift_reconvert < error_rate_q31).all()
        assert (wav_float_shift- wav_q7_shift_reconvert < error_rate_q7).all()

        print("Shift Pass !")

        # Windowing
        wav_q15_windowing = converter_q15.windowing(wav_q15)
        wav_q31_windowing = converter_q31.windowing(wav_q31)
        wav_q7_windowing = converter_q7.windowing(wav_q7)
        wav_float_windowing  = converter_float.windowing(wav_float)

        wav_q15_windowing_reconvert = converter_q15.reconvert(wav_q15_windowing)
        wav_q31_windowing_reconvert = converter_q31.reconvert(wav_q31_windowing)
        wav_q7_windowing_reconvert = converter_q7.reconvert(wav_q7_windowing)

        assert (wav_float_windowing- wav_q15_windowing_reconvert< error_rate_q15).all()
        assert (wav_float_windowing- wav_q31_windowing_reconvert < error_rate_q31).all()
        assert (wav_float_windowing- wav_q7_windowing_reconvert < error_rate_q7).all()

        print("Windowing Pass !")

        # Cosine
        wav_q15_cos = converter_q15.cos(wav_q15[2, 0])
        wav_q31_cos = converter_q31.cos(wav_q31[2, 0])
        # wav_q7_cos = converter_q7.cos(wav_q7[2, 0])
        wav_float_cos  = converter_float.cos(wav_float[2, 0])
        
        wav_q15_cos_reconvert = (converter_q15.reconvert(np.array([wav_q15_cos, ], dtype=wav_q15.dtype)))[0]
        wav_q31_cos_reconvert = (converter_q31.reconvert(np.array([wav_q31_cos, ], dtype=wav_q31.dtype)))[0]
        
        assert (wav_float_cos- wav_q15_cos_reconvert< error_rate_q15).all(), f"Error: {wav_float_cos- wav_q15_cos_reconvert}"
        assert (wav_float_cos- wav_q31_cos_reconvert< error_rate_q31).all(), f"Error: {wav_float_cos- wav_q31_cos_reconvert}"

        print("Cosine Pass !")

        # FFT
        wav_q15_fft = converter_q15.fft(wav_q15)
        wav_q31_fft = converter_q31.fft(wav_q31)
        # wav_q7_fft = converter_q7.fft(wav_q7)
        wav_float_fft  = converter_float.fft(wav_float)
            
        wav_q15_fft_reconvert = converter_q15.reconvert(wav_q15_fft)
        wav_q31_fft_reconvert = converter_q31.reconvert(wav_q31_fft)
        # wav_q7_fft_reconvert = converter_q7.reconvert(wav_q7_fft)

        print("rfft result !")
        print("\tError Max in fft Q15: ", np.max(wav_float_fft-wav_q15_fft_reconvert))
        print("\tError Max in fft Q31: ", np.max(wav_float_fft-wav_q31_fft_reconvert))
        
        # Magnitude
        wav_q15_fft_amp = converter_q15.mag(wav_q15_fft)
        wav_q31_fft_amp = converter_q31.mag(wav_q31_fft)
        # wav_q7_fft_amp = converter_q7.mag(wav_q7)
        wav_float_fft_amp  = converter_float.mag(wav_float_fft)
    
        wav_q15_fft_amp_reconvert = converter_q15.reconvert(wav_q15_fft_amp)
        wav_q31_fft_amp_reconvert = converter_q31.reconvert(wav_q31_fft_amp)
        # wav_q7_fft_amp_reconvert = converter_q7.reconvert(wav_q7_fft)

        print("rfft abs result !")
        print("\tError Max in fft abs Q15: ", np.max(np.abs(wav_q15_fft_amp_reconvert-wav_float_fft_amp)))
        print("\tError Max in fft abs Q31: ", np.max(np.abs(wav_q31_fft_amp_reconvert-wav_float_fft_amp)))

        # # iFFT
        wav_q15_ifft = converter_q15.ifft(wav_q15_fft)
        wav_q31_ifft = converter_q31.ifft(wav_q31_fft)
        # wav_q7_ifft = converter_q7.ifft(wav_q7)
        wav_float_ifft  = converter_float.ifft(wav_float_fft)

        wav_q15_ifft_reconvert = converter_q15.reconvert(wav_q15_ifft)
        wav_q31_ifft_reconvert = converter_q31.reconvert(wav_q31_ifft)
        # wav_q7_ifft_reconvert = converter_q7.reconvert(wav_q7_ifft)
        
        print("rifft result !")
        if wav_q15_ifft.shape != wav_float_ifft.shape:
            warnings.warn(f"Q15, Q31 inverse transform length {wav_q15_ifft_reconvert.shape} is shorter than original {wav_float_ifft.shape}")
            print("\tError Max in fft Q15 -> ifft Q15: ", np.max(np.abs(wav_float_ifft[...,:wav_q15_ifft_reconvert.shape[-1]]-wav_q15_ifft_reconvert)))
            print("\tError Max in fft Q31 -> ifft Q31: ", np.max(np.abs(wav_float_ifft[...,:wav_q31_ifft_reconvert.shape[-1]]-wav_q31_ifft_reconvert)))
        else:
            print("\tError Max in fft Q15 -> ifft Q15: ", np.max(np.abs(wav_float_ifft-wav_q15_ifft_reconvert)))
            print("\tError Max in fft Q31 -> ifft Q31: ", np.max(np.abs(wav_float_ifft-wav_q31_ifft_reconvert)))