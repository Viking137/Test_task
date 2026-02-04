import numpy as np
from scipy.integrate import solve_ivp

from src.data_models import SimulationInput, SpringParameters, StateVector
from src.orbital_mechanics import convert_keplerian_to_cartesian, EARTH_GRAVITATIONAL_PARAMETER

def _calculate_spring_force(
    r_sc: np.ndarray, r_lv: np.ndarray, spring: SpringParameters
) -> np.ndarray:
    """
    Calculates the force vector of the spring acting on the spacecraft.
    The force is considered zero if the spring has reached its final length.
    """
    l_natural_m = spring.natural_length_mm / 1000.0
    l_final_m = spring.final_length_mm / 1000.0

    r_rel = r_sc - r_lv
    distance = np.linalg.norm(r_rel)

    if distance >= l_final_m:
        return np.array([0.0, 0.0, 0.0])

    force_magnitude = spring.stiffness_Npm * (l_natural_m - distance)
    force_vector = force_magnitude * (r_rel / distance)

    return force_vector


def _ode_system(
    t: float,
    state: np.ndarray,
    m_sc: float,
    m_lv: float,
    spring: SpringParameters,
    mu: float,
) -> np.ndarray:
    """
    The right-hand side of the ODE system for the solve_ivp solver.
    Describes the derivatives of the state vector.
    state = [r_sc_x, r_sc_y, r_sc_z, v_sc_x, v_sc_y, v_sc_z,
             r_lv_x, r_lv_y, r_lv_z, v_lv_x, v_lv_y, v_lv_z]
    """
    r_sc, v_sc = state[0:3], state[3:6]
    r_lv, v_lv = state[6:9], state[9:12]

    f_spring_on_sc = _calculate_spring_force(r_sc, r_lv, spring)

    r_sc_norm = np.linalg.norm(r_sc)
    f_gravity_on_sc = -mu * m_sc * r_sc / (r_sc_norm**3)

    r_lv_norm = np.linalg.norm(r_lv)
    f_gravity_on_lv = -mu * m_lv * r_lv / (r_lv_norm**3)

    a_sc = (f_gravity_on_sc + f_spring_on_sc) / m_sc
    a_lv = (f_gravity_on_lv - f_spring_on_sc) / m_lv  

    return np.concatenate([v_sc, a_sc, v_lv, a_lv])


def run_simulation(sim_input: SimulationInput) -> object:
    """
    Runs a full simulation of the separation process.

    1. Calculates the initial conditions for the spacecraft and the PH stage.
    2. Calls scipy.integrate.solve_ivp to solve the ODE system.
    3. Returns the solution object from solve_ivp.
    """
    cm_initial_state = convert_keplerian_to_cartesian(sim_input.orbit)
    r_cm_0 = cm_initial_state.position_m
    v_cm_0 = cm_initial_state.velocity_mps

    l_initial_m = sim_input.spring.initial_length_mm / 1000.0
    r_rel_0 = l_initial_m * (v_cm_0 / np.linalg.norm(v_cm_0))

    m_sc = sim_input.spacecraft_mass_kg
    m_lv = sim_input.launch_vehicle_mass_kg
    m_total = m_sc + m_lv

    r_sc_0 = r_cm_0 + r_rel_0 * (m_lv / m_total)
    r_lv_0 = r_cm_0 - r_rel_0 * (m_sc / m_total)

    v_sc_0 = v_cm_0
    v_lv_0 = v_cm_0

    y0 = np.concatenate([r_sc_0, v_sc_0, r_lv_0, v_lv_0])

    ode_args = (m_sc, m_lv, sim_input.spring, EARTH_GRAVITATIONAL_PARAMETER)
    
    solution = solve_ivp(
        fun=_ode_system,
        t_span=(0, sim_input.simulation_time_s),
        y0=y0,
        args=ode_args,
        dense_output=True,  
        rtol=1e-9,          
        atol=1e-12,  
        max_step=0.01        
    )

    return solution
