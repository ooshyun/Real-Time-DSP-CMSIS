import os
import numpy as np
import src.audio as audio
import unittest

# import warnings
# from numpy.testing import assert_allclose

Amplitude = -40 # dBFS
DURATION = 400 # msec
PATH_TEST_SINE_INT16_TONE = f"./test/data/sin_1k_tone_16bit_{str(DURATION)}ms_{str(Amplitude)}dBFS.wav"
PATH_TEST_STEREO_SINE_INT16_TONE = f"./test/data/sin_1k_stereo_tone_16bit_{str(DURATION)}ms_{str(Amplitude)}dBFS.wav"

PATH_TEST_SINE_INT24_TONE = f"./test/data/sin_1k_tone_24bit_{str(DURATION)}ms_{str(Amplitude)}dBFS.wav"
PATH_TEST_STEREO_SINE_INT24_TONE = f"./test/data/sin_1k_stereo_tone_24bit_{str(DURATION)}ms_{str(Amplitude)}dBFS.wav"

PATH_TEST_SINE_INT32_TONE = f"./test/data/sin_1k_tone_32bit_{str(DURATION)}ms_{str(Amplitude)}dBFS.wav"
PATH_TEST_STEREO_SINE_INT32_TONE = f"./test/data/sin_1k_stereo_tone_32bit_{str(DURATION)}ms_{str(Amplitude)}dBFS.wav"

class AudioSanityCheck(unittest.TestCase):
    def test_make_tone(self):
        amplitude = 1
        duration = 1
        freuquency = 1000
        phase = 0.
        sample_rate = 44100
        type = "sin"
        sine_tone = audio.make_tone(amplitude=amplitude, 
                                duration=duration, 
                                frequency=freuquency, 
                                phase=phase,
                                sample_rate=sample_rate,
                                type=type)
        
        nframe = 512
        nfft = 1024
        fft = np.fft.fft(sine_tone[:nframe], n=nfft)
        target_frequency = (np.abs(freuquency-np.arange(nframe)*44100/nfft))
        target_frequency = np.where(target_frequency==np.min(target_frequency))
        
        assert np.abs(fft)[target_frequency]-nframe//2 < 1e-1

    def test_write_audio(self):
        """
        python -m unittest -v test.test_audio.AudioSanityCheck.test_write_audio
        """
        sine_tone = audio.make_tone(amplitude=10**(Amplitude/20), 
                                    duration=DURATION/1000, 
                                    frequency=1000, 
                                    phase=0.,
                                    sample_rate=44100,
                                    type="sin")
        sine_stereo_tone = np.stack([sine_tone, sine_tone], axis=-1)

        audio.write_audio(path = PATH_TEST_STEREO_SINE_INT32_TONE,
                          data=sine_tone,
                          sample_rate=44100,
                          subtype="PCM_32",
                          )
        audio.write_audio(path = PATH_TEST_STEREO_SINE_INT32_TONE,
                          data=sine_stereo_tone,
                          sample_rate=44100,
                          subtype="PCM_32",
                          )
        
    def test_read_audio(self):
        """
        python -m unittest -v test.test_audio.AudioSanityCheck.test_read_audio
        """
        wav, sr = audio.read_audio(
                        path=PATH_TEST_SINE_INT16_TONE,
                        sample_rate=44100)
        print(wav.dtype, wav.shape, sr)

    def test_convert_audio_channels(self):
        """
        python -m unittest -v test.test_audio.AudioSanityCheck.test_convert_audio_channels
        """
        wav, sr = audio.read_audio(path=PATH_TEST_SINE_INT16_TONE,
                        sample_rate=44100)
        
        wav_mono_ch = audio.convert_audio_channels(wav, 1)
        assert wav_mono_ch.shape[-1] == 1
        
        wav_stereo_ch = audio.convert_audio_channels(wav, 2)
        assert wav_stereo_ch.shape[-1] == 2
        
        wav_spatial_ch = audio.convert_audio_channels(wav, 3)
        assert wav_spatial_ch.shape[-1] == 3
                
        wav_test = audio.convert_audio_channels(wav_stereo_ch, 1)
        assert wav_test.shape[-1] == 1

        wav_test = audio.convert_audio_channels(wav_stereo_ch, 2)
        assert wav_test.shape[-1] == 2
        
        wav_test = audio.convert_audio_channels(wav_spatial_ch, 1)
        assert wav_test.shape[-1] == 1
        
        wav_test = audio.convert_audio_channels(wav_spatial_ch, 2)
        assert wav_test.shape[-1] == 2
    
        wav_test = audio.convert_audio_channels(wav_spatial_ch, 3)
        assert wav_test.shape[-1] == 3
        
    def test_read_file_stream(self):
        """
        python -m unittest -v test.test_audio.AudioSanityCheck.test_read_file_stream
        """
        in_streamer = audio.get_in_streamer(sample_rate=44100,
                            file_path=PATH_TEST_SINE_INT16_TONE,
                            type="file")

        in_streamer.start()

        first = True
        num_frames = 1
        
        frame_length = 512
        stride = int(frame_length//2)
        total_length = frame_length*2

        time = 0
        zero_count = 0
        while (zero_count < 10):
            try:
                length = total_length if first else stride
                first = False
                frame, overflow = in_streamer.read(length)
                
                print(length, frame[..., :3, :], frame.shape, overflow)
                # underflow = stream_out.write(frame) # numpy array
                if np.all(frame[..., 0] == 0):
                    zero_count +=1
                time +=1
            except KeyboardInterrupt:
                print("Stopping")
                break            

        in_streamer.stop()

    def test_write_file_stream(self):
        """
        python -m unittest -v test.test_audio.AudioSanityCheck.test_write_file_stream
        """
        outfile_path = "./test/data/test_write_file_stream.wav"
        in_streamer = audio.get_in_streamer(sample_rate=44100, 
                            file_path=PATH_TEST_SINE_INT16_TONE,
                            type="file")
        out_streamer = audio.get_out_streamer(sample_rate=44100,
                            file_path=outfile_path,
                            type="file")

        in_streamer.start()
        out_streamer.start()

        first = True
        num_frames = 1
        
        frame_length = 512
        stride = int(frame_length//2)
        total_length = frame_length*2

        time = 0
        zero_count = 0
        while (zero_count < int(44100//stride)):
            try:
                length = total_length if first else stride
                first = False
                frame, overflow = in_streamer.read(length)
                
                print(frame[:3, :], frame.shape, overflow)
                underflow = out_streamer.write(frame) # numpy array
                if np.all(frame[..., 0] == 0):
                    zero_count +=1
                time +=1
            except KeyboardInterrupt:
                print("Stopping")
                break            

        in_streamer.stop()
        out_streamer.stop()

    def test_get_mic_streamer(self):
        """
        python -m unittest -v test.test_audio.AudioSanityCheck.test_get_mic_streamer
        """
        outfile_path = "./test/data/test_get_mic_streamer.wav"
        
        in_streamer = audio.get_in_streamer(sample_rate=44100, 
                                            device=2,
                                            type="device")
        
        out_streamer = audio.get_out_streamer(sample_rate=44100,
                            file_path=outfile_path,
                            type="file")

        in_streamer.start()
        out_streamer.start()

        first = True
        num_frames = 1
        
        frame_length = 512
        stride = int(frame_length//2)
        total_length = frame_length*2

        time = 0
        zero_count = 0
        while (zero_count < int(44100//stride)):
            try:
                length = total_length if first else stride
                first = False
                frame, overflow = in_streamer.read(length)
                
                print(zero_count, frame[:3, ...], frame.shape, overflow)
                underflow = out_streamer.write(frame) # numpy array
                if np.all(frame[..., 0] == 0):
                    zero_count +=1
                time +=1
            except KeyboardInterrupt:
                print("Stopping")
                break            

        in_streamer.stop()
        out_streamer.stop()

    def test_get_spk_streamer(self):
        """
        python -m unittest -v test.test_audio.AudioSanityCheck.test_get_spk_streamer
        """
        in_streamer = audio.get_in_streamer(sample_rate=44100, 
                            file_path=PATH_TEST_SINE_INT16_TONE,
                            type="file")
        out_streamer = audio.get_out_streamer(sample_rate=44100,
                                              device=6, # check port using sounddevice.query_device
                                              type="device",
                                              )
        in_streamer.start()
        out_streamer.start()

        first = True
        num_frames = 1
        
        frame_length = 512
        stride = int(frame_length//2)
        total_length = frame_length*2

        time = 0
        zero_count = 0
        while (zero_count < int(44100//stride)):
            try:
                length = total_length if first else stride
                first = False
                frame, overflow = in_streamer.read(length)
                print(zero_count, frame[:3, ...], frame.shape, frame.dtype, overflow)
                underflow = out_streamer.write(frame) # numpy array
                if np.all(frame[..., 0] == 0):
                    zero_count +=1
                time +=1
            except KeyboardInterrupt:
                print("Stopping")
                break            

        in_streamer.stop()
        out_streamer.stop()
        
    def test_print_device_info(self):
        """
        python -m unittest -v test.test_audio.AudioSanityCheck.test_print_device_info
        """
        print()
        print(audio.device_info())

    def test_query_devices(self):
        """
        python -m unittest -v test.test_audio.AudioSanityCheck.test_query_devices
        """
        print(audio.query_devices(device="eqMac", kind="input"))
        print(audio.query_devices(device="eqMac", kind="output"))
        print(audio.query_devices(device="BlackHole 2ch", kind="input"))
        
        print("-"*30)
        x = audio.device_info()
        for info in x:
            print(info)
        
    def test_convert_wav_and_pcm(self):
        """
        python -m unittest -v test.test_audio.AudioSanityCheck.test_convert_wav_and_pcm
        """
        import scipy.io.wavfile as wav
        wav_path = (PATH_TEST_SINE_INT16_TONE, 
                    PATH_TEST_STEREO_SINE_INT16_TONE, 
                    PATH_TEST_SINE_INT32_TONE,
                    PATH_TEST_STEREO_SINE_INT32_TONE,
        )
        sample_rate = (44100, )*4
        channel = (1, 2, 1, 2)
        bitdepth = (16, 16, 32, 32)
        
        type = (
                "ffmpeg", 
                "python",
                )
        pcm_path = "./test/data/test.pcm"
        result_wav_path = "./test/data/test.wav"


        print("\n"+"-"*20)
        for tp in type:
            for path, sr, ch, dtype in zip(wav_path, sample_rate, channel, bitdepth):
                print("Case: ", tp, path, sr, ch, dtype)
                audio.convert_wav_to_pcm(wav_path=path, 
                                        pcm_path=pcm_path,
                                        sample_rate=sr,
                                        channel=ch,
                                        bit_depth=dtype,
                                        type=tp)
                
                audio.convert_pcm_to_wav(pcm_path=pcm_path,
                                        wav_path=result_wav_path, 
                                        sample_rate=sr,
                                        channel=ch,
                                        bit_depth=dtype)
                sr_signal, signal = wav.read(result_wav_path)

                if ch==1:
                    assert len(signal.shape) == 1, f"{signal.shape}"
                else:
                    assert signal.shape[-1] == ch, f"{signal.shape[-1]}, {ch}"
                
                if signal.dtype == np.int16:
                    assert dtype == 16, f"{signal.dtype}, {dtype}"

                if signal.dtype == np.int32:
                    assert dtype == 32, f"{signal.dtype}, {dtype}"

                assert sr_signal == sr, f"{sr_signal}, {sr}"

                print("Pass")
                print("-"*20)
        

        os.remove(pcm_path)
        os.remove(result_wav_path)

    def test_verify_bit_depth(self):
        """
        python -m unittest -v test.test_audio.AudioSanityCheck.test_verify_bit_depth
        """        
        import scipy.io.wavfile as wav
        # path = "./test/data/WhiteNoise_Full_-40_dBFS_44k_PCM16.wav"
        # path = PATH_TEST_STEREO_SINE_INT16_TONE
        path = PATH_TEST_SINE_INT24_TONE
        sr_signal, signal = wav.read(path)

        print(path, ": Signal Info: ", signal.dtype, signal.shape, sr_signal)

        if signal.dtype == np.int16:
            bit_depth=16
        elif signal.dtype == np.int32:
            bit_depth=32
    
        audio.convert_wav_to_pcm(wav_path=path, 
                                #  pcm_path="./test/data/WhiteNoise_Full_-40_dBFS_44k_PCM16.pcm",
                                # pcm_path= f"./test/data/sin_1k_stereo_tone_16bit_{str(DURATION)}ms_{str(Amplitude)}dBFS.pcm",
                                pcm_path= f"./test/data/sin_1k_tone_24bit_{str(DURATION)}ms_{str(Amplitude)}dBFS.pcm",
                                sample_rate=sr_signal,
                                channel=signal.shape[-1] if len(signal.shape)>=2 else 1,
                                bit_depth=bit_depth,
                                type="ffmpeg")

