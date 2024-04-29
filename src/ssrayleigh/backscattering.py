"""
Implementation of Rayleigh backscattering in optical fiber
"""
import numpy as np

from light_source import LightSource
from optical_fiber import OpticalFiber

def backscatterer(source: LightSource, fiber: OpticalFiber):
    """
    Compute Rayleigh backscattering of given source in given fiber.
    """
    # Compute propagated phase
    phases = np.zeros((fiber.number_of_segments + 1, source.angular_wavenumbers.size))
    phases[1:] += np.cumsum(fiber.refractive_indices, axis=0)[:,None]
    phases[1:,:] *= source.angular_wavenumbers[None,:] * fiber.segment_sizes[:,None]


    previous_phase = np.exp(1j * 2 * phases[:-1])

    #* 2 * np.exp(-α * z) * np.sinh(d / 2 * (α + 1j * 2 * n_bar * k)) / (α + 1j * 2 * n_bar * k)