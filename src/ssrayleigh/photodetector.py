"""
Implementation of Photodetector
"""
from dataclasses import dataclass

import numpy as np

def moving_average(array: np.ndarray, n):
    """Compute moving average with given window size over array"""
    ret = np.cumsum(array, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

@dataclass
class Photodector:
    """
    Photodector object with optical and numerical parameters.

    Parameters:
        responsitivity: Responsivity of photodetector, in A/W
        bandwidth: Electrical bandwidth of 
    """
    responsitivity: float
    bandwidth: float 

    def detect(self, fields: np.ndarray, segment_duration: float) -> np.ndarray:
        """
        Detect given field, converting to intensity, then to current.
        Bandwidth is modelled as a moving average.

        Parameters:
            fields: Array of backscattered EM field of shape (N, M) for N scattering zones and M wavelengths
            segment_duration: Time per scattering zone, in s

        Returns:
            Array of current detected for each scattering zones

        Notes:
            Implementation detailed in 
                A. Masoudi et T. P. Nweson, Analysis of distributed optical fibre 
                acoustic sensors through numerical modelling
        """
        bandwidth_points = int(1 / (segment_duration * self.bandwidth))
        intensity = (np.abs(fields) ** 2).sum(axis=1)
        current = self.responsitivity * intensity
        
        return moving_average(current, n=bandwidth_points)

    