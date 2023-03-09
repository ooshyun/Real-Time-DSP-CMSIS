"""
def func(frame, format_converter: ConverterFloatToQFormat)
"""
import numpy as np
from .dsp import ConverterFloatToQFormat
from omegaconf import OmegaConf

from .nr import NRVAD, NRSpectralGate

EPS_NUMPY = np.finfo(np.float32).eps

class NALNL2():
    ...

def dummy_process_fft(frame):
    return frame