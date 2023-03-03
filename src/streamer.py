import tqdm
import warnings
import numpy as np
from omegaconf import DictConfig

from .audio import (
    get_in_streamer, 
    get_out_streamer,
)

from .dsp import ConverterFloatToQFormat


def directionality_time(frame):
    if len(frame.shape) > 1 and frame.shape[-1] > 1:
        frame = np.mean(frame, axis=-1, keepdims=True)

    return frame

def record_fft(frame):
    return frame


class Streamer(object):
    def __init__(self, config: DictConfig):
        self.pre_time_func = [
                            # directionality_time, 
                            ]
        self.freq_func = [
                            record_fft, 
                        ]
        self.post_time_func = []

        self.config = config
        self.dsp_converter = ConverterFloatToQFormat(**config.params)

    def loop(self):
        print("\tPress Ctrl + C if want to stop...")

        assert self.config.params.overlap in (50, 25), f"Overlap can support only 25%, 50%"
        sample_rate = self.config.params.sample_rate
        duration = self.config.params.duration
        frame_length = self.config.params.window_size
        overlap = self.config.params.overlap
        stride = int(frame_length*overlap/100)
        Qformat = self.config.params.qformat
        
        first  = False
        frame_count = 0

        format_converter = self.dsp_converter 
        pre_time_func_list = self.pre_time_func
        freq_func_list = self.freq_func
        post_time_func_list = self.post_time_func
        
        pre_process_time = False if len(pre_time_func_list) == 0 else True
        process_freq = False if len(freq_func_list) == 0 else True
        post_process_time = False if len(post_time_func_list) == 0 else True

        if Qformat == 15:
            dtype = np.int16
        elif Qformat == 31:
            dtype = np.int32
        else:
            dtype = np.float32

        in_streamer = get_in_streamer(**self.config.setting)
        out_streamer = get_out_streamer(**self.config.setting)

        if overlap == 25:
            window_sum = np.ones(shape=(frame_length, 2), dtype=np.float32)*2/3
            window_sum = format_converter.convert(window_sum)

        if process_freq:
            pre_overlap_frame_buffer = np.zeros(shape=(frame_length, 2), dtype=dtype)
            post_overlap_frame_buffer = np.zeros(shape=(frame_length, 2), dtype=dtype)


        if Qformat:
            fft_qformat = int(np.log2(frame_length))
            # ifft_window = format_converter.shift(format_converter.window, -fft_qformat)
            # window_sum = format_converter.shift(window_sum, -fft_qformat)
        else:
            fft_qformat = None
            
        in_streamer.start()
        out_streamer.start()

        total_frame = int((duration*sample_rate)//stride) if duration else False

        warnings.warn(f"There's no setting to stop a record...")
        tmp = []
        while(frame_count < total_frame if total_frame else True):
            try:
                frame, overflow = in_streamer.read(stride)

                print(f"\t Input frame {frame.shape, frame.dtype} Stride {stride}, {frame[4:8, 0]}, {overflow}")
                
                frame = format_converter.convert(frame)

                print(f"\t Format cvt frame {frame.shape, frame.dtype}, {frame[4:8, 0]}")

                if first:
                    if frame.shape[-1] > 2: warnings.warn(f"Currently supported until stereo channel")

                if pre_process_time:
                    for time_func in pre_time_func_list:
                        frame = time_func(frame)

                print(f"\t pre_process_time frame {frame.shape, frame.dtype}, {frame[4:8, 0]}")

                if process_freq:
                    # overlap and add
                    pre_overlap_frame_buffer = np.roll(pre_overlap_frame_buffer, -stride, axis=-2)
                    pre_overlap_frame_buffer[..., -stride:, :] = frame
                    
                    # fft
                    fft_frame = format_converter.fft(format_converter.windowing(pre_overlap_frame_buffer))

                    print(f"\t process_freq fft frame {fft_frame.shape, fft_frame.dtype}, {fft_frame[4:8, 0]}")

                    # fft process
                    for freq_func in freq_func_list:
                        fft_frame = freq_func(fft_frame) 

                    # overlap and add
                    ifft_frame = format_converter.ifft(fft_frame)
                
                    if not first:
                        post_overlap_frame_buffer = np.roll(post_overlap_frame_buffer, -stride, axis=-2)
                        post_overlap_frame_buffer[..., -stride:, :] = 0
                    
                    overlap_buffer = format_converter.windowing(frame=ifft_frame)
                    if overlap == 50:
                        # [TODO] Normalized window
                        raise NotImplementedError
                    if overlap == 25:
                        overlap_buffer = format_converter.mult(overlap_buffer, window_sum) # Q1.15, underflow in here
                        
                    post_overlap_frame_buffer += overlap_buffer
                                    
                    frame = post_overlap_frame_buffer[..., :stride, :]
                        
                    print(f"\t After process_freq ifft frame {ifft_frame.shape, ifft_frame.dtype}, {ifft_frame[4:8, 0]}")
                    print(f"\t After process_freq frame {frame.shape, frame.dtype}, {frame[4:8, 0]}")


                if post_process_time:
                    for time_func in post_process_time:
                        frame = time_func(frame)

                print(f"\t After post process frame {frame.shape, frame.dtype}, {frame[4:8, 0]}")

                frame = format_converter.reconvert(frame)

                print(f"\t After reconversion frame {frame.shape, frame.dtype}, {frame[4:8, 0]}")

                underflow = out_streamer.write(frame)

                print(f"\t Underflow {underflow}, frame {frame_count} total_frame {total_frame}")
                
                frame_count +=1
                first = False
                print(f"-"*30)

                # if frame_count == 30:break
                
            except KeyboardInterrupt:
                print("Stopping")
                break      

        in_streamer.stop()
        out_streamer.stop()

        # file = self.config.setting.in_file.split("/")[-1]
        # # file = "./test/result/Hifi2-mini-vs-CMSIS/CMSIS/fft/out_FFT_Int32_" + file.split(".")[0] + ".npy"
        # file = "./test/result/Hifi2-mini-vs-CMSIS/CMSIS/fft/out_FFT_Float_" + file.split(".")[0] + ".npy"
        # print(file)
        # tmp = np.concatenate(tmp, axis=0)
        # np.save(file, arr=tmp, allow_pickle=True)
        