import unittest
import warnings
from src.streamer import Streamer
from omegaconf import OmegaConf
from src.audio import device_info
class StreamerSanityCheck(unittest.TestCase):
    def test_streamer(self):
        """
        python -m unittest -v test.test_streamer.StreamerSanityCheck.test_streamer
        """
        print(device_info())

        config = OmegaConf.load("./test/conf/config.yaml")
        config.params.duration = 0.5
        # config.setting.in_device = 3

        config.setting.in_type = "file"
        config.setting.in_file = "./data/audio/wav/sin_1k_stereo_tone_16bit_400ms_-40dBFS.wav"

        config.setting.out_type = "file"
        config.setting.out_file = f"./test/data/test_streamer_in_{config.setting.in_type}_out_{config.setting.out_type}_{config.setting.sample_rate}.wav"

        streamer = Streamer(config)

        streamer.loop()