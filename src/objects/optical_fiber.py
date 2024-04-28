"""
Implementation of OpticalFiber
"""

from typing import Literal
from dataclasses import dataclass
from functools import cache

import numpy as np

from scipy import constants, stats
from matplotlib import pyplot as plt


@dataclass
class OpticalFiber:
    # TODO: Add method to extrapolate refractive index at other wavelengths
    # TODO: Add method to extrapolate fiber attenuation coefficient at other wavelengths
    """
    OpticalFiber object with optical and numerical parameters.
    Values should be for small wavelength range that is similar for all.

    Parameters:
        refractive_index: Average refractive index of the fiber
        refractive_index_variation: Refractive index variation, this is multiplied by a uniform random variable in the interval [0, 1]
        attenuation_coefficient: Fiber attenuation coefficient, in dB/km
        segment_size: Average size of individual segements
        segment_size_variation: Segment size variation, in +- %
        mode_field_diameter: Effective mode area of single mode fiber, in um^2
        average_backscattering_field_coefficient: Coefficient of the field scattered backwards, inside the numerical aperture
        length: Fiber length in m
        seend: Seed to generate random values from
    """
    refractive_index: float
    refractive_index_variation: float
    attenuation_coefficient: float
    segment_size: float
    segment_size_variation: float
    mode_field_diameter: float
    average_backscattering_field_coefficient: float
    length: float
    seed: float | None

    def __post_init__(self):
        # Initialize RNG with the given seed
        self._random_generator = np.random.default_rng(self.seed)

    @property
    def number_of_segments(self) -> int:
        """Number of fiber segments"""
        return int(self.length / self.segment_size)

    @cache
    @property
    def segments(self) -> np.ndarray:
        """Size of each fiber segment. This property is cached to ensure the value stays identical."""
        # Assume evenly distributed
        segment_locations = np.ones(self.number_of_segments)

        # Apply variation
        segment_variation = (
            self._random_generator(self.number_of_segments) - 0.5
        ) * 2  # Between -1, 1
        segment_variation = segment_variation * self.segment_size_variation / 100
        segment_locations = segment_locations + segment_variation

        return segment_locations * self.segment_size

    @property
    def segment_locations(self) -> np.ndarray:
        """Location of each fiber segment"""
        return np.cumsum(self.segments)

    @cache
    @property
    def refractive_indices(self) -> np.ndarray:
        """Refractive index of each fiber segment"""
        return (
            self.refractive_index
            + self.refractive_index_variation
            * self._random_generator(self.number_of_segments)
        )


if __name__ == "__main__":
    optical_fiber = OpticalFiber
