"""
Implementation of optical modulation by creating and applying pulse.
"""
import numpy as np
import torch

from matplotlib import pyplot as plt
from scipy import signal
from torchaudio import transforms


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

from scipy import signal
from torch import fft

def apply_pulse(impulse_response: np.ndarray, pulse: np.ndarray, axes: int = 0, gpu: bool = False) -> np.ndarray:
    """
    Apply pulse to impulse response of LTI system, by computing a convolution over given axis

    Parameters:
        impulse_response: Impulse response of LTI system, shape (N, M)
        pulse: Weights of given pulse shape and width, shape (N)
        axis: Axis to apply convolution over
        gpu: If True, computes using gpu
    """
    if not gpu:
        return signal.fftconvolve(impulse_response, pulse[:,None], axes=axes, mode="same")
     
    else:
        if not torch.cuda.is_available():
            raise NotImplementedError("GPU computing is only implemented using CUDA.")
        
        raise NotImplementedError("Correct cropping of output array is not implented yet.")

        with torch.no_grad():
            # Move numpy arrays to torch
            impulse_response = torch.from_numpy(impulse_response)
            pulse = torch.from_numpy(pulse)

            # Compute size of FFT output 
            size = impulse_response.size(0) + pulse.size(0) - 1

            # Find next power of 2 for speedup
            fsize = 2 ** np.ceil(np.log2(size)).astype(int)

            # Move arrays to CUDA
            impulse_response = impulse_response.cuda()
            pulse = pulse.cuda()

            # Compute FFT of pulse, a real signal, and broadcast to impulse response shape
            fpulse = torch.fft.fft(pulse, n=fsize).unsqueeze(-1)
            fimpulse_response = torch.fft.fft(impulse_response, dim=0, n=fsize)

            # Apply convolution
            fresult = fpulse * fimpulse_response

            # Get result in time domain, croped to orginal size
            result = torch.fft.ifft(fresult, dim=0)[0:size, :]

            return result.cpu().numpy()



if __name__ == "__main__":
    pulse_width = 1e-9                  # 1 ns
    segment_duration = 1e-11
    rise_time = pulse_width / 10

    time = get_pulse_time_array(pulse_width, segment_duration, rise_time)

    rec_pulse = get_rectangular_pulse(pulse_width, time)
    sine_pulse = get_sine_rectangular_pulse(pulse_width, time, rise_time)

    plt.plot()
    plt.plot(time * 1e9, rec_pulse, label="Impulsion rectangulaire")
    plt.plot(time * 1e9, sine_pulse, label="Impulsion rectangulaire à pente sinusoïdale")
    plt.xlabel("Temps [ns]")
    plt.ylabel("Amplitude [-]")
    plt.legend()
    plt.show()