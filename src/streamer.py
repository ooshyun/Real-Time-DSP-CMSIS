import tqdm
import time
import warnings
import numpy as np
from omegaconf import DictConfig

from .audio import (
    get_stream_in, 
    get_stream_out,
    convert_audio_channels,
)

from .dsp import ConverterFloatToQFormat

from .time_process import(
    directionality_time,
    apply_gain,
)

from .frequency_process import(
    dummy_process_fft,
    NRVAD,
    NRSpectralGate,
)

import logging
import logging.config
import yaml

with open('./src/conf/debug.yaml', 'r') as f:
    log_config = yaml.safe_load(f.read())
    log_config['handlers']['file']['filename'] = log_config['handlers']['file']['root'] + f'/{__name__.replace(".", "_")}.log'
    del log_config['handlers']['file']['root']
    logging.config.dictConfig(log_config)
    
logger = logging.getLogger(__name__)

class Streamer(object):
    def __init__(self, config: DictConfig):

        self.config = config

        self.dsp_converter = ConverterFloatToQFormat(**config.params)

        self.pre_time_func = [
                            # directionality_time, 
                            ]
        
        self.nr_vad = NRVAD(config=config, 
                             qconverter=self.dsp_converter)
        self.nr_spectralgate = NRSpectralGate(config, qconverter=self.dsp_converter)

        self.freq_func = [
                            # dummy_process_fft, 
                            # self.nr_vad.forward,
                            self.nr_spectralgate.forward,
                            # apply_gain(gain=12, nsample=config.params.window_size+2, qconverter=self.dsp_converter),
                        ]
        
        self.post_time_func = [
            # apply_gain(gain=9, qconverter=self.dsp_converter),
            # convert_audio_channels,
        ]

        self.flag = False


    def loop(self):
        logger.info("\tPress Ctrl + C if want to stop...")

        assert self.config.params.overlap in (50, 25), f"Overlap can support only 25%, 50%"
        sample_rate = self.config.params.sample_rate
        duration = self.config.params.duration
        frame_length = self.config.params.window_size
        overlap = self.config.params.overlap
        stride = int(frame_length*overlap/100)
        Qformat = self.config.params.qformat
        
        first = False
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

        stream_in = get_stream_in(**self.config.setting)
        stream_out = get_stream_out(**self.config.setting)

        if overlap == 25:
            window_sum = np.ones(shape=(frame_length, 1), dtype=np.float32)*2/3
            window_sum = format_converter.convert(window_sum)

        if process_freq:
            pre_overlap_frame_buffer = np.zeros(shape=(frame_length, 1), dtype=dtype)
            post_overlap_frame_buffer = np.zeros(shape=(frame_length, 1), dtype=dtype)
            
        stream_in.start()
        stream_out.start()

        if duration > 0:
            total_frame = int((duration*sample_rate)//stride)
        elif duration < 0 and self.config.setting.in_type == "file":
            total_frame = int((stream_in.length)//stride)
        else:
            total_frame = False


        sr_ms = sample_rate / 1000
        stride_ms = stride / sr_ms
        
        total_time = 0
        current_time = 0
        frame_count_time = 0
        last_log_time = 0
        log_delta = 10
        
        if total_frame: logger.info(f"Ready to process audio, total lag: {duration/ sample_rate / 1000:.1f}ms.")
        if not total_frame: warnings.warn(f"There's no setting to stop a record...")

        while(frame_count < total_frame if total_frame else True):
            try:
                begin = time.time()

                if current_time > last_log_time + log_delta:
                    last_log_time = current_time
                    tpf = total_time / frame_count_time * 1000  # sec to ms
                    rtf = tpf / stride_ms                       # frame per stride
                    logger.info(f"time per frame: {tpf:.1f}ms, RTF: {rtf:.1f}")
                    total_time = 0
                    frame_count_time = 0
                                    
                current_time += stride_ms # length / model.sample_rate                                   
                
                frame, overflow = stream_in.read(stride)

                logger.debug(f"\t Input frame {frame.shape, frame.dtype} Stride {stride}, {frame[4:8, 0]}, {overflow}")
                
                frame = format_converter.convert(frame)

                logger.debug(f"\t Format cvt frame {frame.shape, frame.dtype}, {frame[4:8, 0]}")

                if first:
                    if frame.shape[-1] > 2: warnings.warn(f"Currently supported until stereo channel")

                if pre_process_time:
                    for time_func in pre_time_func_list:
                        frame = time_func(frame)

                logger.debug(f"\t pre_process_time frame {frame.shape, frame.dtype}, {frame[4:8, 0]}")

                if process_freq:
                    # overlap and add
                    pre_overlap_frame_buffer = np.roll(pre_overlap_frame_buffer, -stride, axis=-2)
                    pre_overlap_frame_buffer[..., -stride:, :] = frame
                    
                    # fft
                    fft_frame = format_converter.fft(format_converter.windowing(pre_overlap_frame_buffer))

                    logger.debug(f"\t process_freq fft frame {fft_frame.shape, fft_frame.dtype}, {fft_frame[4:8, 0]}")

                    # fft process
                    for freq_func in freq_func_list:
                        fft_frame = freq_func(fft_frame) 

                    if self.flag:
                        frame = fft_frame[..., :stride, :]
                    else:
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
                            
                        logger.debug(f"\t After process_freq ifft frame {ifft_frame.shape, ifft_frame.dtype}, {ifft_frame[4:8, 0]}")
                        logger.debug(f"\t After process_freq frame {frame.shape, frame.dtype}, {frame[4:8, 0]}")


                if post_process_time:
                    for time_func in post_time_func_list:
                        frame = time_func(frame)

                logger.debug(f"\t After post process frame {frame.shape, frame.dtype}, {frame[4:8, 0]}")

                frame = format_converter.reconvert(frame)

                logger.debug(f"\t After reconversion frame {frame.shape, frame.dtype}, {frame[4:8, 0]}")

                underflow = stream_out.write(frame)

                logger.info(f"\t Underflow {underflow}, frame {frame_count} total_frame {total_frame}")
                # logger.info(f"\t Underflow {underflow}, frame {frame_count} total_frame {total_frame}")
                logger.debug(f"-"*30)
            
                frame_count +=1
                total_time += time.time() - begin
                frame_count_time += 1
                
                first = False
                # if frame_count == 30:break
                
            except KeyboardInterrupt:
                logger.info("Stopping")
                break      

        stream_in.stop()
        stream_out.stop()

        # for key, values in self.nr_func.profile.items():
        #     logger.info(f"\t {key}")
        #     logger.info(f"{len(values)}")
        