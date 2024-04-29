"""
Implementation of optical modulation by creating and applying pulse.
"""
import numpy as np
import torch

from matplotlib import pyplot as plt
from scipy import ndimage
from torch.nn import functional


def get_pulse_time_array(
    pulse_width: float, segment_time: float, time_padding: float = 0.0
) -> np.ndarray:
    """
    Compute time array for given pulse width and segment time, padding wach side by given amount.
    """
    return np.arange(-time_padding, pulse_width+time_padding+segment_time, segment_time)


def get_rectangular_pulse(pulse_width: float, time: np.ndarray) -> np.ndarray:
    """
    Compute rectangular pulse enveloppe.
    Pulse will greater than half height from 0 to pulse_width.

    Parameters:
        pulse_width: Length of pulse at half height, in s
        time: Time array, in s

    Returns:
        Array of square pulse weights
    """
    pulse = np.zeros_like(time)
    pulse[(time >= 0) & (time < pulse_width)] = 1.

    return pulse


def get_sine_rectangular_pulse(
    pulse_width: float, time: np.ndarray, rise_time: float
) -> np.ndarray:
    """
    Create a rectangular pulse with slopes given by a sine function.
    Pulse will greater than half height from 0 to pulse_width.

    Parameters:
        pulse_width: Length of pulse at half height, in s
        segment_duration: Duration of a fiber segment, in s
        rise_time: Rise time of the slope, in s

    Returns:
        Array of square pulse weights

    Notes:
        Implementation detailed in 
            L. B. Liokumovich et al., Fundamentals of Optical Fiber Sensing Schemes Based on 
            Coherent Optical Time Domain Reflectometry: Signal Model Under Static Fiber Conditions. 
    """
    if rise_time > pulse_width:
        raise NotImplementedError
    
    # Half height of sine function is at half way
    plateau_width = pulse_width - rise_time

    conditions = [
        time <= -rise_time / 2, 
        (time >= -rise_time / 2) & (time < rise_time / 2) ,
        (time >= rise_time / 2) & (time < (pulse_width - rise_time / 2)),
        (time >= (pulse_width - rise_time / 2)) & (time < (pulse_width + rise_time / 2)) ,
        time >= pulse_width + rise_time / 2,
        ]

    funcs = [
        lambda t: 0,
        lambda t: (np.sin(np.pi * t / rise_time) + 1) / 2,
        lambda t: 1,
        lambda t: (np.sin(np.pi * (t - plateau_width) / rise_time) + 1) / 2,
        lambda t: 0
    ]

    return np.piecewise(time, conditions, funcs)

def apply_pulse(impulse_response: np.ndarray, pulse: np.ndarray, axis: int = 0, gpu: bool = False):
    """
    Apply pulse to impulse response of LTI system, by computing a convolution over given axis

    Parameters:
        impulse_response: Impulse response of LTI system, shape (N, M)
        pulse: Weights of given pulse shape and width, shape (N)
        axis: Axis to apply convolution over
        gpu: If True, computes using gpu
    """
    if not gpu:
        return ndimage.convolve1d(impulse_response, pulse, axis)  
     
    else:
        if not torch.cuda.is_available():
            raise NotImplementedError("GPU computing is only implemented using CUDA.")
        
        with torch.no_grad():
            # Move numpy arrays to CUDA
            impulse_response = torch.from_numpy(impulse_response).cuda()
            pulse = torch.from_numpy(pulse)
            pulse = torch.complex(pulse, torch.zeros_like(pulse)).cuda()

            # PyTorch expects tensors in the format (batch, channel, length)
            impulse_response = impulse_response.unsqueeze(1)
            pulse = pulse.unsqueeze(-1).unsqueeze(-1)

            return functional.conv1d(impulse_response, pulse).cpu().numpy().sum(axis=1) * 1e-2



if __name__ == "__main__":
    pulse_width = 50e-9
    segment_duration = 1e-12
    rise_time = 10e-9

    time = get_pulse_time_array(pulse_width, segment_duration, rise_time)

    rec_pulse = get_rectangular_pulse(pulse_width, time)
    sine_pulse = get_sine_rectangular_pulse(pulse_width, time, rise_time)

    plt.plot()
    plt.plot(time * 1e9, rec_pulse, label="Rectangular pulse")
    plt.plot(time * 1e9, sine_pulse, label="Sine-sloped rectangular pulse")
    plt.xlabel("Time [ns]")
    plt.ylabel("Amplitude [-]")
    plt.show()
