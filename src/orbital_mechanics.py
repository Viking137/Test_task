import numpy as np
from scipy.optimize import newton
from .data_models import OrbitalElements, StateVector
import math

EARTH_GRAVITATIONAL_PARAMETER = 3.986004418e14

def convert_keplerian_to_cartesian(elements: OrbitalElements) -> StateVector:
    """
    Converts the Keplerian elements of the orbit into a Cartesian vector of state (ECI).

    Accepts Keplerian elements and returns the position vector [x, y, z] in meters
    and the velocity vector [vx, vy, vz] in m/s in the inertial coordinate system
    (Earth-Centered Inertial, ECI).

    Features of processing circular orbits (e=0):
    For circular orbits, the pericenter argument (argument_of_periapsis_deg) becomes
    undefined. By convention, it should be set to 0, and then
    mean_anomaly_deg will be counted from the ascending node.

    Args:
        elements: The OrbitalElements object containing the Keplerian elements of the orbit.

    Returns:
        A StateVector object with position_m (np.ndarray) and velocity_mps (np.ndarray).
    """
    a_m = elements.semi_major_axis_km * 1000  
    e = elements.eccentricity
    i_rad = np.deg2rad(elements.inclination_deg)
    raan_rad = np.deg2rad(elements.raan_deg)
    arg_p_rad = np.deg2rad(elements.argument_of_periapsis_deg)
    m_rad = np.deg2rad(elements.mean_anomaly_deg)


    def kepler_eq(eccentric_anomaly: float) -> float:
        """Auxiliary function for the Kepler equation."""
        return eccentric_anomaly - e * np.sin(eccentric_anomaly) - m_rad

    # Handling the case when e ≈ 0 (circular orbit)
    if e < 1e-9: 
        e_rad = m_rad
    else:
        e_rad = newton(kepler_eq, m_rad) # Initial approximation E ~ M

    
    nu_rad = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(e_rad / 2),
        np.sqrt(1 - e) * np.cos(e_rad / 2)
    )

    
    p_semi_latus_rectum = a_m * (1 - e**2)
    r_norm = p_semi_latus_rectum / (1 + e * np.cos(nu_rad))

    position_pqw = np.array([r_norm * np.cos(nu_rad), r_norm * np.sin(nu_rad), 0.0])

    velocity_magnitude_factor = np.sqrt(EARTH_GRAVITATIONAL_PARAMETER / p_semi_latus_rectum)
    velocity_pqw = np.array([
        -velocity_magnitude_factor * np.sin(nu_rad),
        velocity_magnitude_factor * (e + np.cos(nu_rad)),
        0.0
    ])

    cos_raan, sin_raan = np.cos(raan_rad), np.sin(raan_rad)
    cos_arg_p, sin_arg_p = np.cos(arg_p_rad), np.sin(arg_p_rad)
    cos_i, sin_i = np.cos(i_rad), np.sin(i_rad)

    rotation_matrix_pqw_to_eci = np.array([
        [
            cos_raan * cos_arg_p - sin_raan * sin_arg_p * cos_i,
            -cos_raan * sin_arg_p - sin_raan * cos_arg_p * cos_i,
            sin_raan * sin_i,
        ],
        [
            sin_raan * cos_arg_p + cos_raan * sin_arg_p * cos_i,
            -sin_raan * sin_arg_p + cos_raan * cos_arg_p * cos_i,
            -cos_raan * sin_i,
        ],
        [
            sin_arg_p * sin_i,
            cos_arg_p * sin_i,
            cos_i,
        ],
    ])

    position_eci = rotation_matrix_pqw_to_eci @ position_pqw
    velocity_eci = rotation_matrix_pqw_to_eci @ velocity_pqw

    return StateVector(position_m=position_eci, velocity_mps=velocity_eci)

def convert_true_to_mean_anomaly(true_anomaly_deg: float, eccentricity: float) -> float:
    """
    Converts a true anomaly to an average anomaly.

    Args:
        true_anomaly_deg: The true anomaly is in degrees.
        eccentricity: The eccentricity of the orbit.

    Returns:
        The average anomaly is in degrees.
    """
    nu_rad = np.deg2rad(true_anomaly_deg)
    e = eccentricity

    e_rad = 2 * np.arctan2(
        np.sqrt(1 - e) * np.sin(nu_rad / 2),
        np.sqrt(1 + e) * np.cos(nu_rad / 2)
    )

    m_rad = e_rad - e * np.sin(e_rad)

    return np.rad2deg(m_rad)