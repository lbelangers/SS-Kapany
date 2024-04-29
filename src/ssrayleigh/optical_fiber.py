"""
Implementation of OpticalFiber
"""

import math

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from matplotlib import pyplot as plt
from scipy import constants


@dataclass
class OpticalFiber:
    # TODO: Add method to extrapolate refractive index at other wavelengths
    # TODO: Add method to extrapolate fiber attenuation coefficient at other wavelengths
    """
    OpticalFiber object with optical and numerical parameters.
    All values should be for the same reference wavelength.

    Parameters:
        refractive_index: Average refractive index of the fiber
        refractive_index_variation: Refractive index variation, this is multiplied by a uniform random variable in the interval [0, 1]
        attenuation_coefficient: Fiber attenuation coefficient, in dB/km
        segment_size: Average size of individual segements
        segment_size_variation: Segment size variation, in +- %
        mode_field_diameter: Effective mode area of single mode fiber, in um^2
        average_backscattering_field_coefficient: Coefficient of the field scattered backwards, inside the numerical aperture, in um^-2
        length: Fiber length in m
        seend: Seed to generate random values from

    Notes: 
        Implementation detailed in 
            P. Tovar et al., Modelling Intensity Fluctuations of Rayleigh Backscattered Coherent Light in Single-Mode Fiber 
    """
    refractive_index: float
    refractive_index_variation: float
    attenuation_coefficient: float
    segment_size: float
    segment_size_variation: float
    mode_field_diameter: float
    average_backscattering_field_coefficient: float
    length: float
    seed: float | None = None

    def __post_init__(self):
        # Initialize RNG with the given seed
        self._generator = np.random.default_rng(self.seed)

    @property
    def segment_duration(self) -> float:
        """Duration of a segment, in s"""
        return self.refractive_index * self.segment_size / constants.c

    @property
    def linear_attenuation_coefficient(self) -> float:
        """Fiber attenuation coefficient, in m^-1"""
        return self.attenuation_coefficient / (10 * np.log10(math.e)) * 1e-3

    @property
    def number_of_segments(self) -> int:
        """Number of fiber segments"""
        return int(self.length / self.segment_size)

    @cached_property
    def segment_sizes(self) -> np.ndarray:
        """Size of each fiber segment. This property is cached to ensure the value stays identical."""
        # Assume evenly distributed
        segment_locations = np.ones(self.number_of_segments)

        # Apply variation
        segment_variation = (
            self._generator.random(self.number_of_segments) - 0.5
        ) * 2  # Between -1, 1
        segment_variation = segment_variation * self.segment_size_variation / 100
        segment_locations = segment_locations + segment_variation

        return segment_locations * self.segment_size

    @property
    def segment_locations(self) -> np.ndarray:
        """Location of each fiber segment"""
        return np.cumsum(self.segment_sizes)

    @cached_property
    def refractive_indices(self) -> np.ndarray:
        """Refractive index of each fiber segment"""
        index_variation = self.refractive_index_variation * self._generator.random(
            self.number_of_segments
        )
        return self.refractive_index + index_variation


if __name__ == "__main__":
    fiber = OpticalFiber(
        refractive_index=1.44,
        refractive_index_variation=1e-7,
        attenuation_coefficient=0.2,  # 0.2 dB / km
        segment_size=1e-2,  # 1 cm
        segment_size_variation=5,  # 5 %
        mode_field_diameter=2.13,
        average_backscattering_field_coefficient=70,
        length=10e3,  # 10 km
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, layout="constrained", sharey=True)

    ax1.hist(fiber.segment_sizes * 1e2)
    ax1.set_xlabel("Segment size [cm]")
    ax1.set_ylabel("Occurence [-]")

    ax2.hist(fiber.refractive_indices)
    ax2.set_xlabel("Refractive index [-]")

    plt.show()
