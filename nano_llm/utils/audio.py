#!/usr/bin/env python3
import math
import logging

import torch
import torchaudio

import numpy as np
import pyaudio as pa

from .tensor import convert_dtype, convert_tensor, convert_dtype


def convert_audio(samples, dtype=np.int16):
    """
    Convert between audio datatypes like float<->int16 and apply sample re-scaling.
    If the samples are a raw bytes array, it's assumed that they are in int16 format.
    Supports audio samples as byte buffer, numpy ndarray, and torch.Tensor.  Converted
    byte buffers will be returned as ndarray, otherwise the same object type as input.
    """
    if isinstance(samples, bytes):
        if isinstance(dtype, torch.dtype):
            samples = torch.frombuffer(samples, dtype=torch.int16)
        else:
            samples = np.frombuffer(samples, dtype=np.int16)
    elif not isinstance(samples, (np.ndarray, torch.Tensor)):
        raise TypeError(f"samples should either be bytes, np.ndarray, or torch.Tensor (was {type(samples)})")
        
    if samples.dtype == dtype:
        return samples

    def is_float(dtype):
        return (dtype == torch.float32 or dtype == torch.float64 or dtype == np.float32 or dtype == np.float64)
        
    if is_float(samples.dtype):
        rescale_dtype = dtype
    else:
        rescale_dtype = samples.dtype
        
    #sample_width = np.dtype(str(dtype).split('.')[-1]).itemsize
    sample_width = np.dtype(convert_dtype(rescale_dtype, to='np')).itemsize
    max_value = float(int((2 ** (sample_width * 8)) / 2) - 1)  # 32767 for 16-bit

    if isinstance(samples, np.ndarray):
        numpy_dtype = convert_dtype(dtype, to='np')
        if is_float(samples.dtype):  # float-to-int
            samples = samples * max_value
            samples = samples.clip(-max_value, max_value)
            samples = samples.astype(numpy_dtype)
        elif is_float(dtype):  # int-to-float
            samples = samples.astype(numpy_dtype)
            samples = samples / max_value
        else:
            raise TypeError(f"unsupported audio sample dtype={samples.dtype}")
    elif isinstance(samples, torch.Tensor):
        torch_dtype = convert_dtype(dtype, to='pt')
        if is_float(samples.dtype):
            samples = samples * max_value
            samples = samples.clip(-max_value, max_value).type(dtype=torch_dtype)
        elif is_float(dtype):
            samples = samples.to(dtype=torch_dtype) / max_value
        else:
            raise TypeError(f"unsupported audio sample dtype={samples.dtype}")
    
    if isinstance(samples, np.ndarray) and isinstance(dtype, torch.dtype):
        samples = convert_tensor(samples, return_tensors='pt')
    elif isinstance(samples, torch.Tensor) and not isinstance(dtype, torch.dtype):
        samples = convert_tensor(samples, return_tensors='np')
            
    return samples


_resamplers = {}

def resample_audio(samples, orig_freq=16000, new_freq=16000, warn=None):
    """
    Resample audio to a different sampling rate, while maintaining the pitch.
    """
    global _resamplers
    
    if orig_freq == new_freq:
        return samples
    
    return_tensors = 'pt' if isinstance(samples, torch.Tensor) else 'np'
    
    # lookup or create the resampler    
    key = (orig_freq, new_freq)
    
    if key not in _resamplers:
        _resamplers[key] = torchaudio.transforms.Resample(orig_freq, new_freq).cuda()

    samples = convert_tensor(samples, return_tensors='pt', device='cuda')
    type_in = samples.dtype
    samples = convert_audio(samples, dtype=torch.float32)
    samples = _resamplers[key](samples)
    samples = convert_audio(samples, dtype=type_in)
    samples = convert_tensor(samples, return_tensors=return_tensors)
    
    if warn is not None:
        if not hasattr(warn, '_resample_warning'):
            logging.warning(f"{type(warn)} is resampling audio from {orig_freq} Hz to {new_freq} Hz")
            warn._resample_warning = True
        
    return samples
    
    
def audio_rms(samples):
    """
    Compute the average audio RMS (returns a float between 0 and 1)
    """
    if isinstance(samples, torch.Tensor):
        return torch.sqrt(torch.mean(convert_audio(samples, dtype=torch.float32)**2)).item()
    else:
        return np.sqrt(np.mean(convert_audio(samples, dtype=np.float32)**2))


def audio_db(samples):
    """
    Compute RMS of audio samples in dB.
    """
    rms = audio_rms(samples)
    
    if rms != 0.0:
        return 20.0 * math.log10(rms)
    else:
        return -100.0
    
    
def audio_silent(samples, threshold=0.0):
    """
    Detect if the audio samples are silent or muted.
    
    If threshold < 0, false will be returned (silence detection disabled).
    If threshold > 0, the audio's average RMS will be compared to the threshold.
    If threshold = 0, it will check for any non-zero samples (faster than RMS)
    
    Returns true if audio levels are below threshold, otherwise false.
    """
    if threshold < 0:
        return False
        #raise ValueError("silence threshold should be >= 0")
        
    if threshold == 0:
        if isinstance(samples, bytes):
            samples = np.frombuffer(samples, dtype=np.int16)
        nonzero = np.count_nonzero(samples)
        return (nonzero == 0)
    else:       
        return audio_rms(samples) <= threshold
        
        
_audio_device_info = None

def get_audio_devices(audio_interface=None):
    """
    Return a list of audio devices (from PyAudio/PortAudio)
    """
    global _audio_device_info
    
    if _audio_device_info:
        return _audio_device_info
        
    if audio_interface:
        interface = audio_interface
    else:
        interface = pa.PyAudio()
        
    info = interface.get_host_api_info_by_index(0)
    numDevices = info.get('deviceCount')
    
    _audio_device_info = []
    
    for i in range(0, numDevices):
        _audio_device_info.append(interface.get_device_info_by_host_api_device_index(0, i))
    
    if not audio_interface:
        interface.terminate()
        
    return _audio_device_info
    
    
def find_audio_device(device, audio_interface=None):
    """
    Find an audio device by it's name or ID number.
    """
    devices = get_audio_devices(audio_interface)
    
    if device is None:
        device = len(devices) - 1
        logging.warning(f"audio device unspecified, defaulting to id={device} '{devices[device]['name']}'")
        
    try:
        device_id = int(device)
    except ValueError:
        if not isinstance(device, str):
            raise ValueError("expected either a string or an int for 'device' parameter")
            
        found = False
        
        for id, dev in enumerate(devices):
            if device.lower() == dev['name'].lower():
                device_id = id
                found = True
                break
                
        if not found:
            raise ValueError(f"could not find audio device with name '{device}'")
            
    if device_id < 0 or device_id >= len(devices):
        raise ValueError(f"invalid audio device ID ({device_id})")
        
    return devices[device_id]
    
    
def list_audio_inputs():
    """
    Print out information about present audio input devices.
    """
    devices = get_audio_devices()

    print('')
    print('----------------------------------------------------')
    print(f" Audio Input Devices")
    print('----------------------------------------------------')
        
    for i, dev_info in enumerate(devices):    
        if (dev_info.get('maxInputChannels')) > 0:
            print("Input Device ID {:d} - '{:s}' (inputs={:.0f}) (sample_rate={:.0f})".format(i,
                  dev_info.get('name'), dev_info.get('maxInputChannels'), dev_info.get('defaultSampleRate')))
                 
    print('')
    
    
def list_audio_outputs():
    """
    Print out information about present audio output devices.
    """
    devices = get_audio_devices()
    
    print('')
    print('----------------------------------------------------')
    print(f" Audio Output Devices")
    print('----------------------------------------------------')
        
    for i, dev_info in enumerate(devices):  
        if (dev_info.get('maxOutputChannels')) > 0:
            print("Output Device ID {:d} - '{:s}' (outputs={:.0f}) (sample_rate={:.0f})".format(i,
                  dev_info.get('name'), dev_info.get('maxOutputChannels'), dev_info.get('defaultSampleRate')))
                  
    print('')
    
    
def list_audio_devices():
    """
    Print out information about present audio input and output devices.
    """
    list_audio_inputs()
    list_audio_outputs()
    
    
def pyaudio_dtype(format, to='np'):
    """
    Convert the PyAudio formats to 'np' (numpy) or 'pt' (torch) datatypes
    https://github.com/jleb/pyaudio/blob/0109cc46cac6a3c404050f4ba11752e51aeb1fda/src/pyaudio.py#L128
    """
    to_numpy = {
        pa.paFloat32: np.float32,
        pa.paInt32: np.int32,
        pa.paInt16: np.int16,
        pa.paInt8: np.int8,
        pa.paUInt8: np.uint8,
    }
    
    to_torch = {
        pa.paFloat32: torch.float32,
        pa.paInt32: torch.int32,
        pa.paInt16: torch.int16,
        pa.paInt8: torch.int8,
        pa.paUInt8: torch.uint8,
    }
    
    if to == 'np':
        dtype = to_numpy.get(format)
    elif to == 'pt':
        dtype = to_torch.get(format)
    else:
        raise ValueError(f"the 'to' argument should either be 'np' or 'pt' (was '{to}')")
    
    if dtype is None:
        raise ValueError(f"unsupported PyAudio data format: {format}")
        
    return dtype
        
