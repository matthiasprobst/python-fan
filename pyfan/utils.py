"""Utilities"""
from typing import Union

import numpy as np


def dynamic_pressure_from_volume_flowrate(density: float,
                                          cross_section_area: float,
                                          volume_flow_rate: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute the dynamic pressure based on the volume flow rate.
    This is only correct if the velocity has the shape of a block-profile"""
    return density * (volume_flow_rate / cross_section_area) ** 2
