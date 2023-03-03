import numpy as np
import typing as tp
import warnings
from cmsisdsp import(
    # Q15 Math
    arm_float_to_q15,
    arm_q15_to_float,
    arm_shift_q15,
    arm_add_q15,
    arm_sub_q15,
    arm_mult_q15,
    arm_abs_q15,
    # Q15 Fast Math
    arm_divide_q15,
    arm_sqrt_q15,
    arm_cos_q15,
    # Q15 DFT
    arm_rfft_instance_q15,
    arm_rfft_init_q15,
    arm_rfft_q15,
    # arm_cfft_instance_q15,
    # arm_cfft_init_q15,
    # arm_cfft_q15,    
    arm_cmplx_mag_q15,
    # Q31 Math
    arm_float_to_q31,
    arm_q31_to_float,
    arm_shift_q31,
    arm_add_q31,
    arm_sub_q31,
    arm_mult_q31,
    arm_abs_q31,
    # Q31 Fast Math
    arm_divide_q31,
    arm_sqrt_q31,
    arm_cos_q31,
    # Q31 DFT
    arm_rfft_instance_q31,
    arm_rfft_init_q31,
    arm_rfft_q31, # cmsis-dsp/Source/TransformFunctions/ar_rfft_q31.c
    # arm_cfft_instance_q31,
    # arm_cfft_init_q31,
    # arm_cfft_q31,      # also have radix 2, 4 version
    arm_cmplx_mag_q31,
    # Q7 Math
    arm_float_to_q7,
    arm_q7_to_float,
    arm_shift_q7,
    arm_add_q7,
    arm_sub_q7,
    arm_mult_q7,
    arm_abs_q7,
    # Q7 Fast Math
    # arm_divide_q7,
    # arm_cos_q7,
    # Q7 DFT
    # arm_rfft_instance_q7,
    # arm_rfft_init_q7,
    # arm_rfft_q7,
    # arm_cmplx_mag_q7,   
    # Floating Math
    arm_cos_f32,

    # Filtering
    arm_conv_q7,
    arm_conv_q15,
    arm_conv_q31,
)

def get_fft_func_arm(window_size, qformat, ifft_flag):
    bit_reverse_flag = 1
    if qformat == 15:
        fft_function = arm_rfft_instance_q15()
        status = arm_rfft_init_q15(fft_function, window_size, ifft_flag, bit_reverse_flag)        
    elif qformat == 31:
        fft_function = arm_rfft_instance_q31()
        status = arm_rfft_init_q31(fft_function, window_size, ifft_flag, bit_reverse_flag)  
    if status==0:
        return fft_function
    else:
        raise ValueError(f"FFT initialization had an error {status}...")

def get_window_arm(name, window_size, qformat):
    window = np.zeros(window_size)
    if name == "hann":
        for i in range(window_size):
            window[i] = 0.5 * (1 - arm_cos_f32(2 * np.pi * i / window_size ))
    else:
        raise NotImplementedError(f"{name} window is not supported...")

    if qformat == 15:
        window = arm_float_to_q15(window)
    elif qformat == 31:
        window = arm_float_to_q31(window)
    elif qformat == 7:
        window = arm_float_to_q7(window)
    
    return window
        
class ConverterFloatToQFormat(object):
    def __init__(self, 
                window_size,
                qformat=0, # 0: float, others: Q format
                window_type="hann",
                *args,
                **kwargs,
                ):
        assert qformat in (0, 7, 15, 31), f"Q format is possible only q7, q15, q31"

        print(qformat, window_size)

        self.qformat = qformat
        self.window_type = window_type
        self.window_size = window_size
        self.norm_factor = int(window_size//2)
        self.ifft_factor = int(np.log2(window_size))
        self.window = get_window_arm(name=window_type, 
                                    window_size=window_size,
                                    qformat=qformat)
        if qformat == 15:
            self.fft_function = get_fft_func_arm(window_size, qformat, ifft_flag=0)
            self.ifft_function = get_fft_func_arm(window_size, qformat, ifft_flag=1)
        elif qformat == 31:
            self.fft_function = get_fft_func_arm(window_size, qformat, ifft_flag=0)
            self.ifft_function = get_fft_func_arm(window_size, qformat, ifft_flag=1)
        elif self.qformat == 7:
            warnings.warn(f"Q7 format do not implment fft computation yet...")
            self.fft_function = None
            self.ifft_function = None
        else:
            self.fft_function = np.fft.rfft 
            self.ifft_function = np.fft.irfft 

    def set_window_type(self, window_size, window_type):
        self.window_size = window_size
        self.window = get_window_arm(name=window_type, 
                                    window_size=window_size,
                                    qformat=self.qformat)
        self.norm_factor = int(window_size//2)
        self.ifft_factor = int(np.log2(window_size))
        
    def set_qformat(self, qformat):
        self.qformat = qformat

        self.window = get_window_arm(name=self.window_type, 
                                    window_size=self.window_size,
                                    qformat=self.qformat)
        if qformat == 15:
            self.fft_function = get_fft_func_arm(self.window_size, qformat, ifft_flag=0)
            self.ifft_function = get_fft_func_arm(self.window_size, qformat, ifft_flag=0)
        elif qformat == 31:
            self.fft_function = get_fft_func_arm(self.window_size, qformat, ifft_flag=1)
            self.ifft_function = get_fft_func_arm(self.window_size, qformat, ifft_flag=1)   

    def reconvert(self, a: np.ndarray):
        shape = a.shape
        if self.qformat == 15:
            assert a.dtype == np.int16, f"variable should be int16"            
            return arm_q15_to_float(a.flatten()).reshape(shape)
        elif self.qformat == 31:
            assert a.dtype == np.int32, f"variable should be int32"            
            return arm_q31_to_float(a.flatten()).reshape(shape)
        elif self.qformat == 7:
            assert a.dtype == np.int8, f"variable should be int8"
            return arm_q7_to_float(a.flatten()).reshape(shape)
        else:
            return a

    def convert(self, a: np.ndarray):
        if a.dtype == np.float64:
            a = a.astype(np.float32)

        assert a.dtype == np.float32, f"variable should be float32"
        
        shape = a.shape
        if self.qformat == 15:
            return arm_float_to_q15(a.flatten()).reshape(shape)
        elif self.qformat == 31:
            return arm_float_to_q31(a.flatten()).reshape(shape)
        elif self.qformat == 7:
            return arm_float_to_q7(a.flatten()).reshape(shape)
        else:
            return a

    def shift(self, a: np.ndarray, shift: int):
        assert isinstance(shift, int), f"shift factor should be integer"

        warnings.warn("Caraful Shift, Overflow/Underflow operation is not implmented yet...")

        shape = a.shape
        if self.qformat == 15:
            assert a.dtype == np.int16, f"variable should be int16"
            result = arm_shift_q15(a.flatten(), shift).reshape(shape)
        elif self.qformat == 31:
            assert a.dtype == np.int32, f"variable should be int32"
            result = arm_shift_q31(a.flatten(), shift).reshape(shape)
        elif self.qformat == 7:
            assert a.dtype == np.int8, f"variable should be int8"
            result = arm_shift_q7(a.flatten(), shift).reshape(shape)
        else:
            result = a*(2**shift)
        
        return result

    def add(self, a: np.ndarray, b: np.ndarray):
        assert a.shape == b.shape, f"variable shape should be same..."
        shape = a.shape
        if self.qformat:
            if self.qformat == 15:
                assert a.dtype == np.int16 and b.dtype == np.int16, f"variable should be int16"
                return arm_add_q15(a.flatten(), b.flatten()).reshape(shape)
            elif self.qformat == 31:
                assert a.dtype == np.int32 and b.dtype == np.int32, f"variable should be int32"
                return arm_add_q31(a.flatten(), b.flatten()).reshape(shape)
            elif self.qformat == 7:
                assert a.dtype == np.int8 and b.dtype == np.int8, f"variable should be int8"
                return arm_add_q7(a.flatten(), b.flatten()).reshape(shape)
        else:
            return a+b

    def sub(self, a: np.ndarray, b: np.ndarray):
        assert a.shape == b.shape, f"variable shape should be same..."
        shape = a.shape
        if self.qformat:
            if self.qformat == 15:
                assert a.dtype == np.int16 and b.dtype == np.int16, f"variable should be int16"
                return arm_sub_q15(a.flatten(), b.flatten()).reshape(shape)            
            elif self.qformat == 31:
                assert a.dtype == np.int32 and b.dtype == np.int32, f"variable should be int32"
                return arm_sub_q31(a.flatten(), b.flatten()).reshape(shape)
            elif self.qformat == 7:
                assert a.dtype == np.int8 and b.dtype == np.int8, f"variable should be int8"
                return arm_sub_q7(a.flatten(), b.flatten()).reshape(shape)
        else:
            return a-b

    def mult(self, a: np.ndarray, b: np.ndarray):
        assert a.shape == b.shape, f"variable shape should be same..."
        shape = a.shape
        if self.qformat:
            if self.qformat == 15:
                assert a.dtype == np.int16 and b.dtype == np.int16, f"variable should be int16"
                return arm_mult_q15(a.flatten(), b.flatten()).reshape(shape)            
            elif self.qformat == 31:
                assert a.dtype == np.int32 and b.dtype == np.int32, f"variable should be int32"
                return arm_mult_q31(a.flatten(), b.flatten()).reshape(shape)
            elif self.qformat == 7:
                assert a.dtype == np.int8 and b.dtype == np.int8, f"variable should be int8"
                return arm_mult_q7(a.flatten(), b.flatten()).reshape(shape)
        else:
            return a*b

    def abs(self, a: np.ndarray):
        shape = a.shape
        if self.qformat:
            if self.qformat == 15:
                assert a.dtype == np.int16, f"variable should be int16"
                return arm_abs_q15(a.flatten()).reshape(shape)            
            elif self.qformat == 31:
                assert a.dtype == np.int32, f"variable should be int32"
                return arm_abs_q31(a.flatten()).reshape(shape)
            elif self.qformat == 7:
                assert a.dtype == np.int32, f"variable should be int8"
                return arm_abs_q7(a.flatten()).reshape(shape)
        else:
            return np.abs(a)

    def divide(self, a: int, b: int) -> tp.Tuple[int, int, int]:
        """
            return ret, value, shift
        """
        if self.qformat:
            # assert isinstance(a, int) and isinstance(b, int), f"variable should be integer..."
            if self.qformat == 15:
                return arm_divide_q15(a, b)            
            elif self.qformat == 31:
                return arm_divide_q31(a, b)
            elif self.qformat == 7:
                raise NotImplementedError(f"Q7 cannot use division yet...")
                # return arm_divide_q7(a, b)
        else:
            return 0, a/b, 0

    def sqrt(self, a: int) -> tp.Tuple[int, int]:
        """
            return ret, value
        """
        if self.qformat:
            # assert isinstance(a, int), f"variable should be integer..."
            if self.qformat == 15:
                return arm_sqrt_q15(a)            
            elif self.qformat == 31:
                return arm_sqrt_q31(a)
            elif self.qformat == 7:
                raise NotImplementedError(f"Q7 cannot use sqrt yet...")
                return arm_sqrt_q7(a)
        else:
            return 0, np.sqrt(a)

    def cos(self, a: int) -> int:
        # assert isinstance(a, int)

        if self.qformat == 15:
            return arm_cos_q15(a)            
        elif self.qformat == 31:
            return arm_cos_q31(a)
        elif self.qformat == 7:
            raise NotImplementedError(f"Q7 cannot use cosine yet...")
            return arm_cos_q7(a)
        else:
            return np.cos(a)

    def windowing(self, frame:np.ndarray, window:np.ndarray=None):
        assert len(frame.shape) <= 2, "Windowing supports until 2D array, number of sampel, channel..."
        result = np.zeros_like(frame, dtype=frame.dtype)
        
        if window is None:
            _window = self.window
        else:
            _window = window

        if self.qformat == 15:
            assert frame.dtype == np.int16, f"variable should be int16"
            for ch in range(frame.shape[-1]):
                result[..., :, ch] = arm_mult_q15(frame[..., :, ch], _window)
        elif self.qformat == 31:
            assert frame.dtype == np.int32, f"variable should be int32"
            for ch in range(frame.shape[-1]):
                result[..., :, ch] = arm_mult_q31(frame[..., :, ch], _window)
        elif self.qformat == 7:
            assert frame.dtype == np.int8, f"variable should be int8"
            for ch in range(frame.shape[-1]):
                result[..., :, ch] = arm_mult_q7(frame[..., :, ch], _window)
        else:
            result = frame * np.expand_dims(_window, -1)
        
        return result
              
    def fft(self, frame:np.ndarray):
        """
        Return: normalized FFT, Q 1.31
        
        Noramlized value 1 for fft + Q8.24 to Q1.31
        """
        assert len(frame.shape)==2 and frame.shape[-2] == self.window_size, f"Window size should match with data"
        nchannel = frame.shape[-1]
        fft_funcion = self.fft_function
        result = np.zeros(shape=(self.window_size+2, nchannel), dtype=frame.dtype)

        if self.qformat == 15:
            assert frame.dtype == np.int16, f"variable should be int16"
            for ch in range(nchannel):
                result[..., ch] = arm_rfft_q15(fft_funcion, frame[..., ch])[:self.window_size+2] << 1
        elif self.qformat == 31:
            assert frame.dtype == np.int32, f"variable should be int32"
            for ch in range(nchannel):
                result[..., ch] = arm_rfft_q31(fft_funcion, frame[..., ch])[:self.window_size+2] << 1
        elif self.qformat == 7:
            raise NotImplementedError(f"Q7 cannot use fft yet...")
            assert frame.dtype == np.int8, f"variable should be int8"
            for ch in range(nchannel):
                result[..., ch] = arm_rfft_q7(fft_funcion, frame[..., ch])[:self.window_size+2] << 1
        else:
            fft_result = fft_funcion(a=frame, n=self.window_size, axis=-2)
            result[...,0::2, :] = fft_result.real/self.norm_factor
            result[...,1::2, :] = fft_result.imag/self.norm_factor
  
        return result

    def mag(self, frame:np.ndarray):
        """
        Return: Q 1.31 (Q 2.30 -> Q 1.31)
        """
        assert len(frame.shape) == 2, f"Frame for magnitude shape should be (nsample, nchannel)"
        nchannel = frame.shape[-1]
        result = np.zeros(shape=(self.window_size//2+1, nchannel), dtype=frame.dtype)

        if self.qformat == 15:
            assert frame.dtype == np.int16, f"variable should be int16"
            for ch in range(nchannel):
                result[:, ch] = arm_cmplx_mag_q15(frame[..., ch]) << 1
        elif self.qformat == 31:
            assert frame.dtype == np.int32, f"variable should be int32"
            for ch in range(nchannel):
                result[:, ch] = arm_cmplx_mag_q31(frame[..., ch]) << 1
        elif self.qformat == 7:
            raise NotImplementedError(f"Q7 cannot use fft yet...")
            assert frame.dtype == np.int8, f"variable should be int8"
            for ch in range(nchannel):
                result[:, ch] = arm_cmplx_mag_q7(frame[..., ch]) << 1
        else:
            result = np.sqrt(frame[0::2, :]**2 + frame[1::2, :]**2)
        return result

    def ifft(self, fft_frame):
        """
            Noramlized value 1 -> 0.5 for ifft
        """
        assert len(fft_frame.shape)==2 and fft_frame.shape[-2] == (self.window_size//2+1)*2, f"Window size should match with data"
    
        nchannel = fft_frame.shape[-1]

        ifft_function = self.ifft_function
        result = np.zeros(shape=(self.window_size, nchannel), dtype=fft_frame.dtype)
        if self.qformat == 15:
            assert fft_frame.dtype == np.int16, f"variable should be int16"
            for ch in range(nchannel):
                    result[..., ch] = arm_rfft_q15(ifft_function, fft_frame[..., ch]>>1) << self.ifft_factor 

        elif self.qformat == 31:
            assert fft_frame.dtype == np.int32, f"variable should be int32"
            for ch in range(nchannel):
                    result[..., ch] = arm_rfft_q31(ifft_function, fft_frame[..., ch]>>1) << self.ifft_factor 

        elif self.qformat == 7:
            raise NotImplementedError(f"Q7 cannot use ifft yet...")
            assert fft_frame.dtype == np.int8, f"variable should be int8"
            for ch in range(nchannel):
                    result[..., ch]  =  arm_rfft_q7(fft_funcion, fft_frame[..., ch])
        else:
            fft_frame *= self.norm_factor
            result= ifft_function(a=fft_frame[..., 0::2, :] + 1j*fft_frame[..., 1::2, :], 
                                    n=self.window_size, 
                                    axis=-2)
        return result

    def convolve(self, a, b):
        """
        arm_conv_q31
        """
        if self.qformat == 15:
            return arm_conv_q15(a, b)

        elif self.qformat == 31:
            return arm_conv_q31(a, b)
    
        elif self.qformat == 7:
            return arm_conv_q7(a, b)
    
        else:
            return np.convolve(a, b, mode="full", )
        
    def fir(self, frame, coeff):
        """
        arm_fir_q31
        """

    def interpolate(self, frame):
        """
        arm_linear_interp_q31
        arm_bilinear_interp_q31
        """

    def normalized_window(self, frame):
        # TODO for 50 overlap
        raise NotImplementedError
        def window_sumsquare_scratch(window, n_frames, win_length, n_fft, hop_length, dtype, norm=None):
            """ window_sumsquare
                1. getting window
                2. normalizae **2
                3. pad_center depending on fft size
                4. overlap and add(hopping the window and add)
                
            Hypothesis: window length is same as n_fft. If different, it should be centering
            """
            if win_length is None:
                win_length = n_fft

            n = n_fft + hop_length * (n_frames - 1)
            x = np.zeros(n, dtype=dtype)
            # Compute the squared window at the desired length
            win_sq = signal.get_window(window=window, Nx=win_length)
            win_sq = librosa.util.normalize(win_sq, norm=norm) ** 2
            # win_sq = util.pad_center(win_sq, size=n_fft)
            
            # Fill the envelope, __window_ss_fill
            n_fft = len(win_sq)
            for i in range(n_frames):
                sample = i * hop_length
                x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
            return x

        # Normalize by sum of squared window, librosa.filters.window_sumsquare
        ifft_window_sum = window_sumsquare_scratch(
            window=name_window,
            n_frames=num_frame,
            win_length=win_length,
            n_fft=nfft,
            hop_length=hop_length,
            dtype=wav_result.dtype,
        )

        # plt.plot(ifft_window_sum)

        # if hop_length != nfft//4:
        approx_nonzero_indices = ifft_window_sum > librosa.util.tiny(ifft_window_sum)
        wav_result[..., approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]
