from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass(frozen=True)
class OrbitalElements:
    """Keplerian elements of the orbit."""
    semi_major_axis_km: float
    eccentricity: float
    argument_of_periapsis_deg: float
    raan_deg: float  # Right Ascension of the Ascending Node
    inclination_deg: float
    mean_anomaly_deg: float

@dataclass(frozen=True)
class SpringParameters:
    """Parameters of the spring pusher."""
    stiffness_Npm: float      
    initial_length_mm: float
    natural_length_mm: float
    final_length_mm: float

@dataclass(frozen=True)
class SimulationInput:
    """All the input data for the simulation."""
    orbit: OrbitalElements
    spring: SpringParameters
    spacecraft_mass_kg: float
    launch_vehicle_mass_kg: float
    simulation_time_s: float


@dataclass
class StateVector:
    """Representation of the vector of the state of one body in the ECI."""
    position_m: np.ndarray    # [x, y, z] 
    velocity_mps: np.ndarray  # [vx, vy, vz] 


@dataclass
class SimulationResult:
    """A structured representation of the complete simulation results."""
    time_steps_s: np.ndarray             # Array of timestamps
    sc_states: List[StateVector]         # History of spacecraft states
    lv_states: List[StateVector]         # The history of the launch vehicle stage states
    spring_forces_N: np.ndarray          # The history of spring force