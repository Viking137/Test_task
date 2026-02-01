import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np

from src.orbital_mechanics import convert_keplerian_to_cartesian, convert_true_to_mean_anomaly
from src.data_models import OrbitalElements


def test_vallado_example_case():
    """
    Checks the conversion of Keplerian elements to Cartesian coordinates,
    using data from Vallado, Fundamentals of Astrodynamics and Applications (Example 2-3).
    """
    elements = OrbitalElements(
        semi_major_axis_km=36124.678,
        eccentricity=0.83284,
        inclination_deg=87.87,
        raan_deg=227.89,
        argument_of_periapsis_deg=53.38,
        mean_anomaly_deg=7.6056 
    )

    expected_position_m = np.array([6525520.0, 6861470.0, 6449370.0])
    expected_velocity_mps = np.array([4901.78, 5532.74, -1976.13])

    result_state_vector = convert_keplerian_to_cartesian(elements)

    assert result_state_vector.position_m == pytest.approx(expected_position_m, abs=300)
    assert result_state_vector.velocity_mps == pytest.approx(expected_velocity_mps, abs=0.5)
