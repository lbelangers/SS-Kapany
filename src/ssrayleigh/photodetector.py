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

    def get_bandwidth_points(self, segment_duration) -> int:
        """
        Compute bandwidth in points.

        Parameters:
            segment_duration: Time per scattering zone, in s

        Return:
            Number of points for moving average
        """
        return int(1 / (segment_duration * self.bandwidth))

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
        intensity = (np.abs(fields) ** 2).sum(axis=1)
        current = self.responsitivity * intensity
        
        return moving_average(current, n=self.get_bandwidth_points(segment_duration))
    
    def crop_to_bandwidth(self, array: np.ndarray, segment_duration: float) -> np.ndarray:
        """
        Crop array by moving average size.

        Parameters:
            array: Array to crop
            segment_duration: Time per scattering zone, in s

        Returns:
            Cropped array
        """
        N = self.get_bandwidth_points(segment_duration)
        return array[N//2:-(N//2)+ 1 + N % 2]

    