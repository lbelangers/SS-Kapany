"""
Implementation of Rayleigh backscattering in optical fiber
"""

import numpy as np

from light_source import LightSource
from optical_fiber import OpticalFiber

import optical_modulation as omod


def backscatterer(source: LightSource, fiber: OpticalFiber, pulse: np.ndarray):
    """
    Compute Rayleigh backscattering of given source in given fiber.
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

    # Apply electric field strength
    initial_electric_field = source.get_electric_field_amplitude(
        fiber.mode_field_diameter, fiber.refractive_index
    )
    fields *= initial_electric_field * source.normalized_lineshape[None, :]

    # Apply constants
    fields *= fiber.mode_field_diameter * fiber.average_backscattering_field_coefficient

    # Apply pulse 
    fields_probed = omod.apply_pulse(fields, pulse)
