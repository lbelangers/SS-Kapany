"""
Implementation of Rayleigh backscattering in optical fiber
"""

import numpy as np

from light_source import LightSource
from optical_fiber import OpticalFiber

import optical_modulation as omod


def backscatterer(source: LightSource, fiber: OpticalFiber, pulse: np.ndarray):
    """
    Compute Rayleigh backscattering of given source of given pulse in given fiber.

    Parameters:
        source: Light source injected in fiber
        fiber: Fiber where backscattering happens
        pulse: Shape of light pulse
    """
    # Compute propagated phase
    phases = np.zeros((fiber.number_of_segments + 1, source.angular_wavenumbers.size))
    phases[1:] += np.cumsum(fiber.refractive_indices, axis=0)[:, None]
    phases[1:, :] *= source.angular_wavenumbers[None, :] * fiber.segment_sizes[:, None]

    # Compute individual contributions to electric field
    argument = (
        fiber.linear_attenuation_coefficient
        + 1j
        * 2
        * fiber.refractive_indices[:, None]
        * source.angular_wavenumbers[None, :]
    )
    section_contribution = (
        2
        * np.exp(
            -fiber.linear_attenuation_coefficient * fiber.segment_locations[:, None]
        )
        * np.sinh(fiber.segment_sizes[:, None] / 2 * argument)
        / argument
    )
    previous_phase = np.exp(1j * 2 * phases[:-1])
    fields = previous_phase * section_contribution

    # Apply lineshpae
    fields *= source.normalized_lineshape[None,:]

    # Apply electric field
    initial_electric_field = source.get_electric_field_amplitude(fiber.mode_field_diameter, fiber.refractive_index)
    fields *= initial_electric_field

    # Apply constants
    fields *= fiber.mode_field_diameter * fiber.average_backscattering_field_coefficient

    # Applying probe pulse
    fields_probed = omod.apply_pulse(fields, pulse)

    return fields_probed
