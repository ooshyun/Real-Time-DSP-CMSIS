"""
This file is for i/o audio in python. 

Many functions is from denoiser, and it follows denoiser License.

Function List
-------------
read_audio
write_audio

Reference
-------------
Denoiser: https://github.com/facebookresearch/denoiser
Read / Write library performance: https://github.com/faroit/python_audio_loading_benchmark

"""
import os
import ffmpeg
import queue
import librosa
import struct
import numpy as np
from typing import Tuple
import soundfile as sf
import sounddevice as sd
from pathlib import Path
from .utils import bold
import sys

def get_stream_in(sample_rate, in_device=0, in_file="", in_type="file", *args, **kwarg):
    """
    type: "file", "device"  
    """
    if in_type=="file":
        return FileInputStream(file_path=in_file, sample_rate=sample_rate)
    elif in_type=="device":
        return DeviceInputStream(device=in_device, sample_rate=sample_rate)
    else:
        raise ValueError(f"There's no {in_type} type streamer")


def get_stream_out(sample_rate, out_device=0, out_file="", out_type="file", *args, **kwarg):
    """
    type: "file", "device"  
    """
    if out_type=="file":
        return FileOutputStream(file_path=out_file, sample_rate=sample_rate)
    elif out_type=="device":
        return DeviceOutputStream(device=out_device, sample_rate=sample_rate)
    else:
        raise ValueError(f"There's no {out_type} type streamer")


def parse_audio_device(device):
    if device is None:
        return device
    try:
        return int(device)
    except ValueError:
        return device


def query_devices(device, kind):
    try:
        caps = sd.query_devices(device, kind=kind)
    except ValueError:
        message = bold(f"Invalid {kind} audio interface {device}.\n")
        message += (
            "If you are on Mac OS X, try installing Soundflower "
            "(https://github.com/mattingalls/Soundflower).\n"
            "You can list available interfaces with `python3 -m sounddevice` on Linux and OS X, "
            "and `python.exe -m sounddevice` on Windows. You must have at least one loopback "
            "audio interface to use this.")
        print(message, file=sys.stderr)
        sys.exit(1)
    return caps


def device_info():
    return sd.query_devices()
    

def available_subtypes(format=None):
    """
    ------------------------------
    {'PCM_S8': 'Signed 8 bit PCM', 
    'PCM_16': 'Signed 16 bit PCM', 
    'PCM_24': 'Signed 24 bit PCM', 
    'PCM_32': 'Signed 32 bit PCM', 
    'PCM_U8': 'Unsigned 8 bit PCM', 
    'FLOAT': '32 bit float', 
    'DOUBLE': '64 bit float', 
    'ULAW': 'U-Law', 
    'ALAW': 'A-Law', 
    'IMA_ADPCM': 'IMA ADPCM', 
    'MS_ADPCM': 'Microsoft ADPCM', 
    'GSM610': 'GSM 6.10', 
    'G721_32': '32kbs G721 ADPCM', 
    'G723_24': '24kbs G723 ADPCM', 
    'G723_40': '40kbs G723 ADPCM', 
    'DWVW_12': '12 bit DWVW', 
    'DWVW_16': '16 bit DWVW', 
    'DWVW_24': '24 bit DWVW', 
    'VOX_ADPCM': 'VOX ADPCM', 
    'NMS_ADPCM_16': '16kbs NMS ADPCM', 
    'NMS_ADPCM_24': '24kbs NMS ADPCM', 
    'NMS_ADPCM_32': '32kbs NMS ADPCM', 
    'DPCM_16': '16 bit DPCM', 
    'DPCM_8': '8 bit DPCM', 
    'VORBIS': 'Vorbis', 
    'OPUS': 'Opus', 
    'MPEG_LAYER_I': 'MPEG Layer I', 
    'MPEG_LAYER_II': 'MPEG Layer II', 
    'MPEG_LAYER_III': 'MPEG Layer III', 
    'ALAC_16': '16 bit ALAC', 
    'ALAC_20': '20 bit ALAC', 
    'ALAC_24': '24 bit ALAC', 
    'ALAC_32': '32 bit ALAC'
    """
    return sf.available_subtypes(format)


class DeviceInputStream(object):
    def __init__(self, 
                device, 
                sample_rate):
        self.port = parse_audio_device(device)
        self.info = query_devices(device=device, kind="input")
        self.channel =  min(self.info['max_input_channels'], 2)
        
        print(f"**Input Port {device}, Channel {self.channel}")

        self.streamer = sd.InputStream(
        device=self.port,
        samplerate=sample_rate,
        channels=self.channel)

    def start(self):
        self.streamer.start()

    def stop(self):
        self.streamer.stop()

    def read(self, length):
        frame, overflow = self.streamer.read(length)
        if len(frame.shape) == 1:
            frame = np.expand_dims(frame, axis=-1)
        return frame, overflow


class DeviceOutputStream(object):
    def __init__(self, 
                device, 
                sample_rate):
        self.port = parse_audio_device(device)
        self.info = query_devices(device=device, kind="output")
        self.channel =  min(self.info['max_output_channels'], 2)
        
        print(f"\t ** Output Port {device}, Channel {self.channel}")

        self.streamer = sd.OutputStream(
        device=self.port,
        samplerate=sample_rate,
        channels=self.channel)

    def start(self):
        self.streamer.start()

    def stop(self):
        self.streamer.stop()

    def write(self, frame):
        if frame.shape[-1] != self.channel:
            frame = convert_audio_channels(frame, channels=self.channel)
        frame = frame.astype(np.float32) # nsample, channel
        underflow = self.streamer.write(frame)
        return underflow


class FileInputStream(object):
    def __init__(self, file_path, sample_rate):
        self.file_path = file_path
        self.streams = None
        self.length = None
        self.channel = None
        self.sample_rate = sample_rate
        self.offset = 0
        
        self.zero_buffer = None
        
    def start(self):
        streams, self.sample_rate = read_audio(path=self.file_path, 
                                sample_rate=self.sample_rate)
        self.channel = streams.shape[-1] if len(streams.shape)!=1 else 1
        self.streams = convert_audio_channels(streams, channels=self.channel)
        self.length = self.streams.shape[-2]

        # import matplotlib.pyplot as plt
        # plt.plot(self.streams[..., 0])
        
    def stop(self):
        self.streams = None
        self.channel = None
        self.length = None
        self.zero_buffer = None

    def read(self, length):
        self.zero_buffer = get_buffer(buffer=self.zero_buffer, 
                                    shape=(length, self.channel))

        if self.offset > self.length:
            out =  self.zero_buffer
        elif self.offset + length > self.length:
            out = np.concatenate([self.streams[..., self.offset:, :], self.zero_buffer[..., :self.offset + length - self.length, :]],
                                 axis=-2)
        else:
            out = self.streams[..., self.offset:self.offset+length, :]
        self.offset += length
        return out, False # [TODO] Overflow?


class FileOutputStream(object):
    def __init__(self, file_path, sample_rate):
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.offset = 0
        self.buffer = None
        self.size = 0
        
    def start(self):
        self.buffer = queue.SimpleQueue()

    def stop(self):
        size = self.buffer.qsize()
        wav = [None, ]*size
        for i in range(size):
            wav[i] = self.buffer.get_nowait()

        if size == 0:
            wav = np.ones(shape=(1, 1), dtype=np.float32)
        else:
            wav = np.concatenate(wav, axis=-2)

        if wav.shape[0] == 1:
            wav = np.squeeze(wav, axis=-1)

        write_audio(path=self.file_path, 
                    data=wav, 
                    sample_rate=self.sample_rate)
        self.buffer = None
        
    def write(self, frame):
        self.buffer.put(frame)
        return False # [TODO] Underflow?
        

def get_mic_streamer(port, sample_rate) -> sd.InputStream:
    device_in = parse_audio_device(port)
    caps = query_devices(device_in, "input")
    channels_in = min(caps['max_input_channels'], 2)

    print(f"Input Port {device_in}, Channel {channels_in}")
    stream_in = sd.InputStream(
        device=device_in,
        samplerate=sample_rate,
        channels=channels_in)

    return stream_in


def get_spk_streamer(port, sample_rate) -> sd.OutputStream:
    device_out = parse_audio_device(port)
    caps = query_devices(device_out, "output")
    
    channels_out = min(caps['max_output_channels'], 2)

    print(f"Output Port {device_out}, Channel {channels_out}")
    stream_out = sd.OutputStream(
        device=device_out,
        samplerate=sample_rate,
        channels=channels_out)
    
    return stream_out


def read_audio(path:str, sample_rate:int = None) -> Tuple[np.ndarray, int]:
    """If want to read pcm file itself, then use import scipy.io.wavfile as wavfile
        (soundfile use c++ library)
    """    
    # return sf.read(
    #     file=path, 
    #     )
    return librosa.load(
        path=path,
        sr = sample_rate
    )


def write_audio(path:str, data:np.ndarray, sample_rate:int, subtype="PCM_16"):
    if len(data.shape)==2 and data.shape[0]==1:
        data = np.squeeze(data, axis=0)
    sf.write(file=path, 
            data=data, 
            samplerate=sample_rate,
            subtype=subtype,
    )


def convert_audio_channels(wav, channels=2):
    """Convert audio to the given number of channels.""" 
    if len(wav.shape) == 1:
        wav = np.expand_dims(wav, axis=-1)
    
    *shape, length, src_channels = wav.shape
    
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, but the stream have multiple
        # channels, downmix all channels.
        # wav = wav.mean(dim=-2, keepdim=True)
        wav = np.mean(wav, axis=-1, keepdims=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file have
        # one single channel, replicate the audio over all channels.
        wav = np.broadcast_to(wav, shape=shape+[length, channels])
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file have
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav


def get_buffer(shape: tuple, buffer: np.ndarray=None):
    if buffer is None:
        buffer = np.zeros(shape=shape)
    else:
        if buffer.shape[-2] > shape[-2]:
            buffer = buffer[..., :shape[-2], :]
        elif buffer.shape[-2] < shape[-2]:
            buffer = np.concatenate([buffer, np.zeros(shape=shape+[shape[-2]-buffer.shape[-2]])], 
                                    axis=-2)
        else:
            pass
    return buffer

def add_noise(signal:np.ndarray, noise:np.ndarray, snr: int, power: float=2., version: int=0):
    """
    Version 0. SNR base
        https://github.com/Sato-Kunihiko/audio-SNR/blob/master/create_mixed_audio_file.py

    Version 1. Background Noise
    """
    assert noise.shape[0] >= signal.shape[0], f"Noise should be longer than signal..."

    max_dtype = np.finfo(signal.dtype).max
    min_dtype = np.finfo(signal.dtype).min

    start = np.random.randint(0, noise.shape[0]-signal.shape[0])
    noise_divided = noise[start:start+signal.shape[0]]
    
    signal_rms = np.sqrt(np.power(signal, power))
    noise_rms = np.sqrt(np.power(noise_divided, power))

    if version ==0:
        adjusted_noise_rms = signal_rms / 10**(snr/20)
        
        mix = signal + adjusted_noise_rms/noise_rms * noise_divided

        if mix.max(axis=0) > max_dtype or mix.min(axis=0) < min_dtype:
            if mix.max(axis=0) >= abs(mix.min(axis=0)): 
                reduction_rate = max_dtype / mix.max(axis=0)
            else :
                reduction_rate = min_dtype / mix.min(axis=0)
            mix = mix * (reduction_rate)
    elif version == 1:
        mix = np.zeros_like(noise_rms)
        mix[noise_rms > signal_rms] = noise_divided[noise_rms > signal_rms]
        mix[noise_rms <= signal_rms] = signal[noise_rms <= signal_rms] + noise_divided[noise_rms <= signal_rms]
        
    return mix




def make_tone(amplitude:int, duration:int, frequency:int, phase:float = 0, sample_rate:int=44100, type="sin"):
    assert 0 <= phase and phase <= 2*np.pi
    if type=="sin":
        func = np.sin
    elif type=="cos":
        func = np.cos

    return amplitude*func(2*np.pi*frequency*np.arange(int(duration*sample_rate))/sample_rate)

        
def make_wav_format(pcm_data:bytes, ch:int, bit_depth:int, sample_rate:int) -> bytes:
        """make wav header and combine with pcm data
        
        Reference 
        ----------
        - http://soundfile.sapp.org/doc/WaveFormat/
        - struct.pack: https://docs.python.org/3/library/struct.html
        """

        waves = []
        waves.append(struct.pack('<4s', b'RIFF'))    # chunk id
        waves.append(struct.pack('I', 1))            # chunk size
        waves.append(struct.pack('<4s', b'WAVE'))    # format
        waves.append(struct.pack('<4s', b'fmt '))    # subchunk1_id
        waves.append(struct.pack('I', 16))           # subchunk1_size
        # audio_format, channel_cnt, sample_rate, bytes_rate(sr*blockalign:초당 바이츠수), block_align, bps
        waves.append(struct.pack('HHIIHH', 
                                1,                              # audio format
                                ch,                             # channel
                                sample_rate,                    # sample rate
                                sample_rate*ch*bit_depth//8,    # byte_rate = SampleRate * NumChannels * BitsPerSample/8
                                ch*bit_depth//8,                # block align = NumChannels * BitsPerSample/8
                                bit_depth))                     # bit_depth, 8bit, 16bit, 32bit
        waves.append(struct.pack('<4s', b'data'))   # subchunk2 id
        waves.append(struct.pack('I', len(pcm_data))) # subchunk2 size
        waves.append(pcm_data)                        # data
        waves[1] = struct.pack('I', sum(len(w) for w in waves[2:])) # chunk size
        return b''.join(waves)


def read_wave_raw(filename):
    """
    from. https://gist.github.com/chief7/54873e6e7009a087180902cb1f4e27be

    Just pass in a filename and get bytes representation of
    audio data as result
    :param filename: a wave file
    :param rate:
    :return: tuple -> data, #channels, samplerate, datatype (in bits)
    """
    with open(filename, "rb") as wav:
        # RIFF-Header: "RIFF<size as uint32_t>WAVE" (12 bytes)
        riff_header = wav.read(12)
        riff, filesize, wave = struct.unpack("<4sI4s", riff_header)

        assert riff.decode("utf-8") == "RIFF"
        assert wave.decode("utf-8") == "WAVE"

        """
        Format header:
        'fmt ' - 4 bytes
        header length - 4 bytes
        format tag - 2 bytes (only PCM supported here)
        channels - 2 bytes
        sample rate - 4 bytes
        bytes per second - 4 bytes
        block align - 2 bytes
        bits per sample - 2 bytes
        """
        fmt_header = wav.read(24)
        fmt_header_data = struct.unpack("<4sIHHIIHH", fmt_header)
        _, header_len, fmt_tag, nchannels, samplerate, _, _, dtype = fmt_header_data
        assert fmt_tag == 1 # only PCM supported

        """
        Data part
        'data' - 4 bytes, header
        len of data - 4 bytes
        """
        data_header = wav.read(8)
        head, data_len = struct.unpack("<4sI", data_header)

        assert head.decode("utf-8") == "data"

        data = b""

        while True:
            chunk = wav.read(samplerate)
            data += chunk

            if len(chunk) < samplerate or len(data) >= data_len:
                # it's possible to encounter another data section you should handle it
                break

        return data, nchannels, samplerate, dtype


def convert_wav_to_pcm(wav_path:str, pcm_path:str, sample_rate:int = 44100, channel:int =1, bit_depth:int = 16, type:str ="ffmpeg"):
    assert type in ("ffmpeg", "python"), f"Convert wav to pcm supports ffmpeg and python method..."
    assert bit_depth in (16, 24, 32), f"PCM can cover 16, 24, 32 bits..."
    
    if type=="ffmpeg":
        """
        f32le           PCM 32-bit floating-point little-endian
        s32le           PCM signed 32-bit little-endian
        s32be           PCM signed 32-bit big-endian
        s16le           PCM signed 16-bit little-endian
        
        # wav to pcm
        ffmpeg -i .wav -acodec pcm_s16le -f s16le -ac 1 -ar 44100 .pcm

        # pcm to wav
        ffmpeg -f s16le -ar 44100 -ac 1 -i .pcm .wav
        """
        pcm, _ =ffmpeg.input(wav_path,
            ).output('-', format=f's{bit_depth}le', acodec=f'pcm_s{bit_depth}le', ac=channel, ar=str(sample_rate)
            ).overwrite_output(
            ).run(capture_stdout=True)
    elif type=="python":
        pcm, ch, sr, dtype = read_wave_raw(wav_path)
        assert ch==channel and sample_rate== sr and bit_depth == dtype, \
            f"file format is different from params, channel {ch}, sample rate {sr} bit depth {dtype}..."
    
    with open(pcm_path, 'wb+') as file:
        file.write(pcm)    


def convert_pcm_to_wav(pcm_path:str, wav_path:str, sample_rate:int = 44100, channel:int=1, bit_depth:int=16):
    pcm_bytes = Path(pcm_path).read_bytes()
    wav_bytes = make_wav_format(pcm_bytes, ch=channel, bit_depth=bit_depth, sample_rate=sample_rate)
    
    with open(wav_path, 'wb') as file:
        file.write(wav_bytes)