"""
Implementation of LightSource
"""

from typing import Literal
from dataclasses import dataclass

import numpy as np

from scipy import constants, stats
from matplotlib import pyplot as plt


@dataclass
class LightSource:
    """
    LightSource object with optical and numerical parameters.

    Parameters:
        center_frequency: Center frequency in Hz of the light emitted by the source.
        linewidth: FWHM of the light source in Hz, defining the spectral width.
        power: Power of the light source in W.
        step_size: Frequency step size in Hz
        bandwidth: Difference between biggest and smallest frequency in Hz
        shape: Lineshape of the source
    """

    frequency: float
    frequency_linewidth: float
    power: float
    frequency_step: float
    bandwidth: float
    shape: Literal["gaussian", "lorentzian"]

    @property
    def frequencies(self) -> np.ndarray:
        """
        Array of frequencies centered around the center frequency,
        spaced by a given step size, over span of given bandwidth.
        """
        start, stop = -self.bandwidth / 2, self.bandwidth / 2
        return np.arange(start, stop, self.frequency_step) + self.frequency

    @property
    def wavelength(self) -> np.ndarray:
        """
        Center wavelength of the light emitted by the source in m
        """
        return constants.c / self.frequency

    @property
    def wavelength_linewidth(self) -> np.ndarray:
        """
        Linewidth of the source
        """
        upper_wavelength = constants.c / (self.frequency - self.frequency_linewidth / 2)
        lower_wavelength = constants.c / (self.frequency + self.frequency_linewidth / 2)
        return upper_wavelength - lower_wavelength

    @property
    def wavelengths(self) -> np.ndarray:
        """
        Array of wavelengths centered around the center wavelength,
        spaced by a given step size, over span of given bandwidth.
        """
        return constants.c / self.frequencies

    @property
    def angular_wavenumber(self) -> np.ndarray:
        """
        Center angular wavenumber of the light emitted by the source in rad m^-1
        """
        return 2 * np.pi / self.wavelength

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
                self.frequencies, self.frequency, self.frequency_linewidth / 2.335
            )

            return distribution / distribution.sum()

        elif self.shape == "lorentzian":
            distribution = stats.cauchy.pdf(
                self.frequencies, self.frequency, self.frequency_linewidth / 2
            )

            return distribution / distribution.sum()

        else:
            raise NotImplementedError

    def get_coherence_length(self, refractive_index: float = 1) -> float:
        """
        Compute the coherence length of the light source in a material of given
        refractive index.

        Parameters:
            refractive_index: Refractive index of propagating media

        Returns:
            Coherence length in meters
        """
        return self.wavelength**2 / (refractive_index * self.wavelength_linewidth)

    def get_electric_field_amplitude(
        self, area: float, refractive_index: float = 1
    ) -> float:
        """
        Compute the amplitude of the electric field in a given medium.

        Parameters:
            area: Amplitude of electric field
            refractive_index: Refractive index of propagating media

        Returns:
            Amplitude of the electric field
        """
        intensity = self.power / area
        return np.sqrt(
            2 * intensity / (constants.c * constants.epsilon_0 * refractive_index)
        )


if __name__ == "__main__":
    central_frequency = 194e12                                                         # 194 THz
    frequency_linewidth = 3e3                                                          # 3 KHz

    source = LightSource(
        frequency=central_frequency,
        frequency_linewidth=frequency_linewidth,
        power=1e-3,                                                                 # 1 mW = 0 dBm
        frequency_step=250,                                                         # 250 Hz
        bandwidth=8 * frequency_linewidth,
        shape="lorentzian",
    )

    plt.plot(source.frequencies / 1e12, source.normalized_lineshape)
    plt.xlabel("Frequencies [THz]")
    plt.ylabel("Amplitude [a.u.]")
    plt.show()
