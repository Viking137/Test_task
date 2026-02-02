import pytest
import numpy as np
import pandas as pd
from types import SimpleNamespace
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import process_results
from src.data_models import SimulationInput, OrbitalElements, SpringParameters

def test_process_results():
    """
    Checks the process_results function based on simple, predictable data.
    """
    spring = SpringParameters(
        stiffness_Npm=1000.0,
        initial_length_mm=100.0,
        natural_length_mm=120.0, 
        final_length_mm=150.0
    )
    sim_input = SimulationInput(
        orbit=None, 
        spring=spring,
        spacecraft_mass_kg=1.0, 
        launch_vehicle_mass_kg=1.0, 
        simulation_time_s=1.0
    )

    mock_solution = SimpleNamespace()
    mock_solution.t = np.array([0.0, 1.0])
    
    y0 = np.array([0.05, 0, 0, 1, 0, 0, -0.05, 0, 0, -1, 0, 0])
    y1 = np.array([0.065, 0, 0, 2, 0, 0, -0.065, 0, 0, -2, 0, 0])

    mock_solution.y = np.vstack([y0, y1]).T

    df = process_results(mock_solution, sim_input)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2 
    assert list(df.columns) == [
        "time_s",
        "r_sc_x", "r_sc_y", "r_sc_z",
        "v_sc_x", "v_sc_y", "v_sc_z",
        "r_lv_x", "r_lv_y", "r_lv_z",
        "v_lv_x", "v_lv_y", "v_lv_z",
        "relative_distance_m",
        "relative_speed_mps",
        "spring_force_N"
    ]

    assert df.loc[0, "time_s"] == 0.0
    assert df.loc[0, "relative_distance_m"] == pytest.approx(0.1)
    assert df.loc[0, "relative_speed_mps"] == pytest.approx(2.0)
    assert df.loc[0, "spring_force_N"] == pytest.approx(20.0)

    assert df.loc[1, "time_s"] == 1.0
    assert df.loc[1, "relative_distance_m"] == pytest.approx(0.13)
    assert df.loc[1, "relative_speed_mps"] == pytest.approx(4.0)
    assert df.loc[1, "spring_force_N"] == pytest.approx(0.0)

    assert df["time_s"].max() == pytest.approx(sim_input.simulation_time_s)
