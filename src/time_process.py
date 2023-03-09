"""
def func(frame, format_converter: ConverterFloatToQFormat)
"""
import numpy as np
from .dsp import ConverterFloatToQFormat

def apply_gain(gain: int, nsample: int, qconverter: ConverterFloatToQFormat):
    _gain = 10**(gain/20)

    if qconverter.qformat:
        gain_shift = int(np.ceil(np.log2(_gain))) if _gain > 0 else 0
        _gain = _gain*(2**(-gain_shift))

    _gain = np.broadcast_to(_gain, shape=(nsample, 1))
    
    if qconverter.qformat:
        _gain = qconverter.convert(_gain)

    def _apply_gain(frame):        
        frame = qconverter.mult(frame, _gain)

        if qconverter.qformat:
            frame = qconverter.shift(frame, gain_shift)

        return frame
    
    return _apply_gain

def apply_gain_band(gain: np.array, qconverter: ConverterFloatToQFormat):
    """
    freq 125         160         200       250       315         400         500        530          800
    gain 17.601237, 17.920655, 18.738822, 20.665206, 22.591590, 23.803367, 24.001173, 24.198979, 24.396785, 

    freq 1000        1250        1600        2000      2500      3150        4000        5000       6300       8000
    gain 24.594591, 24.292397, 24.668297, 24.476770, 24.285243, 24.093716, 23.902189, 23.710663, 23.726691, 24.051170
    """
    ...

def directionality_time(frame):
    if len(frame.shape) > 1 and frame.shape[-1] > 1:
        frame = np.mean(frame, axis=-1, keepdims=True)
    return frame

class NALR():
    ...

class Compressor():
    ...

class Resample():
    ...
