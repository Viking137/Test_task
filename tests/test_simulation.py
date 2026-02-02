import pytest
import numpy as np
import sys
import os
from scipy.integrate import solve_ivp 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation import run_simulation
from src.data_models import SimulationInput, OrbitalElements, SpringParameters
from src.orbital_mechanics import convert_keplerian_to_cartesian, EARTH_GRAVITATIONAL_PARAMETER

@pytest.fixture
def sample_simulation_input() -> SimulationInput:
    """Typical input data for the simulation."""
    orbit = OrbitalElements(
        semi_major_axis_km=7000.0,
        eccentricity=0.01,
        inclination_deg=51.6,
        raan_deg=45.0,
        argument_of_periapsis_deg=30.0,
        mean_anomaly_deg=0.0
    )
    spring = SpringParameters(
        stiffness_Npm=5000.0,
        initial_length_mm=110.0,
        natural_length_mm=150.0,
        final_length_mm=195.0
    )
    return SimulationInput(
        orbit=orbit,
        spring=spring,
        spacecraft_mass_kg=800.0,
        launch_vehicle_mass_kg=2000.0,
        simulation_time_s=5.0
    )

def test_separation_occurs(sample_simulation_input):
    """
    Checks that the distance between objects has increased after the simulation.
    """
    initial_distance_m = sample_simulation_input.spring.initial_length_mm / 1000.0

    solution = run_simulation(sample_simulation_input)

    final_state = solution.y[:, -1]
    r_sc_final = final_state[0:3]
    r_lv_final = final_state[6:9]

    final_distance = np.linalg.norm(r_sc_final - r_lv_final)

    assert final_distance > initial_distance_m



def test_center_of_mass_conservation(sample_simulation_input):
    """
    Checks that the center of mass of the system is moving in a predictable orbit.,
    since the force of the spring is internal.
    """
    solution = run_simulation(sample_simulation_input)
    
    m_sc = sample_simulation_input.spacecraft_mass_kg
    m_lv = sample_simulation_input.launch_vehicle_mass_kg
    m_total = m_sc + m_lv
    
    final_state_separation = solution.y[:, -1]
    r_sc_final = final_state_separation[0:3]
    r_lv_final = final_state_separation[6:9]
    
    cm_pos_after_separation = (m_sc * r_sc_final + m_lv * r_lv_final) / m_total

    no_spring_input = sample_simulation_input
    
    def single_body_ode(t, state, mu, m):
        r, v = state[0:3], state[3:6]
        r_norm = np.linalg.norm(r)
        accel = -mu * r / (r_norm**3)
        return np.concatenate([v, accel])

    cm_initial_state = convert_keplerian_to_cartesian(sample_simulation_input.orbit)
    y0_cm = np.concatenate([cm_initial_state.position_m, cm_initial_state.velocity_mps])

    sol_cm = solve_ivp(
        fun=single_body_ode,
        t_span=(0, sample_simulation_input.simulation_time_s),
        y0=y0_cm,
        args=(EARTH_GRAVITATIONAL_PARAMETER, m_total),
        rtol=1e-9,
        atol=1e-12,
    )
    
    cm_pos_expected = sol_cm.y[0:3, -1]
    
    assert cm_pos_after_separation == pytest.approx(cm_pos_expected, abs=1e-3)

# --- Tests for the function _calculate_spring_force ---
def test_calculate_spring_force_compressed():
    """Проверяет силу пружины, когда она сжата."""
    spring = SpringParameters(stiffness_Npm=1000.0, initial_length_mm=100.0, natural_length_mm=120.0, final_length_mm=195.0)
    r_sc = np.array([0.05, 0, 0])
    r_lv = np.array([-0.05, 0, 0]) # Distance = 0.1m (100mm), spring compressed by 20mm

    from src.simulation import _calculate_spring_force 
    force = _calculate_spring_force(r_sc, r_lv, spring)
    expected_force = np.array([20.0, 0, 0]) 
    assert np.linalg.norm(force) > 0 
    assert force == pytest.approx(expected_force)
    assert np.allclose(force, expected_force)


def test_calculate_spring_force_natural_length():
    """Checks the strength of the spring when it is in its natural state."""
    spring = SpringParameters(stiffness_Npm=1000.0, initial_length_mm=100.0, natural_length_mm=120.0, final_length_mm=195.0)
    r_sc = np.array([0.06, 0, 0])
    r_lv = np.array([-0.06, 0, 0]) # Distance = 0.12m (120mm), natural length

    from src.simulation import _calculate_spring_force
    force = _calculate_spring_force(r_sc, r_lv, spring)
    expected_force = np.array([0.0, 0.0, 0.0])
    assert np.linalg.norm(force) == 0 
    assert np.allclose(force, expected_force)


def test_calculate_spring_force_stretched():
    """Checks the force of the spring when it is stretched, but not yet in the stop."""
    spring = SpringParameters(stiffness_Npm=1000.0, initial_length_mm=100.0, natural_length_mm=120.0, final_length_mm=195.0)
    r_sc = np.array([0.075, 0, 0])
    r_lv = np.array([-0.075, 0, 0]) # Distance = 0.15m (150mm), stretched
    
    from src.simulation import _calculate_spring_force
    force = _calculate_spring_force(r_sc, r_lv, spring)
    expected_force = np.array([0.0, 0.0, 0.0]) 
    assert np.linalg.norm(force) == 0
    assert np.allclose(force, expected_force)


def test_calculate_spring_force_at_final_stop():
    """Checks the force of the spring when it has reached the final stop."""
    spring = SpringParameters(stiffness_Npm=1000.0, initial_length_mm=100.0, natural_length_mm=120.0, final_length_mm=195.0)
    r_sc = np.array([0.0975, 0, 0])
    r_lv = np.array([-0.0975, 0, 0]) # Distance = 0.195m (195mm), final length

    from src.simulation import _calculate_spring_force
    force = _calculate_spring_force(r_sc, r_lv, spring)
    expected_force = np.array([0.0, 0.0, 0.0])
    assert np.linalg.norm(force) == 0
    assert np.allclose(force, expected_force)


def test_calculate_spring_force_beyond_final_stop():
    """Checks the force of the spring when it has exceeded the end stop."""
    spring = SpringParameters(stiffness_Npm=1000.0, initial_length_mm=100.0, natural_length_mm=120.0, final_length_mm=195.0)
    r_sc = np.array([0.1, 0, 0])
    r_lv = np.array([-0.1, 0, 0]) # Distance = 0.2m (200mm), longer than the final length

    from src.simulation import _calculate_spring_force
    force = _calculate_spring_force(r_sc, r_lv, spring)
    expected_force = np.array([0.0, 0.0, 0.0])
    assert np.linalg.norm(force) == 0 
    assert np.allclose(force, expected_force)

