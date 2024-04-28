"""
Implementation of LightSource
"""

from typing import Literal

import numpy as np

from scipy import constants, stats
from matplotlib import pyplot as plt


class LightSource:

    def __init__(
        self,
        center_frequency: float,
        frequency_linewidth: float,
        power: float,
        frequency_step: float,
        bandwidth: float,
        shape: Literal["gaussian"],
    ) -> None:
        """
        Create instance of LightSource object with optical and numerical parameters.

        Parameters:
            center_frequency: Center frequency in Hz of the light emitted by the source.
            linewidth: Linewidth of the light source in Hz, defining the spectral width.
            power: Power of the light source in W.
            step_size: Frequency step size in Hz
            bandwidth: Difference between biggest and smallest frequency in Hz
            shape: Lineshape of the source
        """
        self.center_frequency = center_frequency
        self.frequency_linewidth = frequency_linewidth
        self.power = power
        self.frequency_step = frequency_step
        self.bandwidth = bandwidth
        self.shape = shape

    @property
    def frequencies(self) -> np.ndarray:
        """
        Array of frequencies centered around the center frequency,
        spaced by a given step size, over span of given bandwidth.
        """
        start, stop = -self.bandwidth / 2, self.bandwidth / 2
        return np.arange(start, stop, self.frequency_step) + self.center_frequency

    @property
    def center_wavelength(self) -> np.ndarray:
        """
        Center wavelength of the light emitted by the source in m
        """
        return constants.c / self.center_frequency

    @property
    def wavelength_linewidth(self) -> np.ndarray:
        """
        Linewidth of the source
        """
        upper_wavelength = constants.c / (
            self.center_frequency - self.frequency_linewidth / 2
        )
        lower_wavelength = constants.c / (
            self.center_frequency + self.frequency_linewidth / 2
        )
        return upper_wavelength - lower_wavelength

    @property
    def wavelengths(self) -> np.ndarray:
        """
        Array of wavelengths centered around the center wavelength,
        spaced by a given step size, over span of given bandwidth.
        """
        return constants.c / self.frequencies

    @property
    def angular_wavenumbers(self) -> np.ndarray:
        """
        Array of angular wavenumbers centered around the center angular wavenumber,
        spaced by a given step size, over span of given bandwidth.
        """
        return 2 * np.pi / self.wavelengths

    @property
    def normalized_lineshape(self) -> np.ndarray:
        """
        Normalized lineshape of light source, same shape as frequencies
        """
        if self.shape == "gaussian":
            distribution = stats.norm.pdf(
                self.frequencies, self.center_frequency, self.frequency_linewidth
            )

            return distribution / distribution.sum()

        else:
            raise NotImplementedError

    def get_coherence_length(self, refractive_index: float = 1) -> float:
        """
        Compute the coherence length of the light source in a material of given
        refractive index.

        Returns:
            Coherence length in meters
        """
        return self.center_wavelength**2 / (
            refractive_index * self.wavelength_linewidth
        )


if __name__ == "__main__":
    central_frequency = 194e12  # 194 THz
    central_wavelength = constants.c / central_frequency
    frequency_step = 2.5e6  # 2.5 MHz
    frequency_bandwidth = 20e6  # 20 MHz
    power = 1e-3  # 1 mW = 0 dBm
    wavelength_linewidth = 10e-12
    frequency_linewidth = constants.c / (
        central_wavelength - wavelength_linewidth / 2
    ) - constants.c / (
        central_wavelength + wavelength_linewidth / 2
    )  # 10 pm

    light_source = LightSource(
        center_frequency=central_frequency,
        frequency_linewidth=frequency_linewidth,
        power=power,
        frequency_step=frequency_step,
        bandwidth=10 * frequency_linewidth,
        shape="gaussian",
    )

    plt.plot(light_source.frequencies / 1e12, light_source.normalized_lineshape)
    plt.xlabel("Frequencies [THz]")
    plt.ylabel("Amplitude [a.u.]")
    plt.show()
