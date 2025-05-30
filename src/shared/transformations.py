import numpy as np
from src.shared.constants import (
    EARTH_EQUATORIAL_RADIUS_KM,
    EARTH_ECCENTRICITY_SQ,
    OMEGA_EARTH_RAD_PER_SEC,
    MU_EARTH_KM3_S2,
)


def lla_to_ecef(lat_deg, lon_deg, alt_km):
    """Converts Latitude, Longitude, Altitude to ECEF coordinates (km)."""
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)

    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)
    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)

    N = EARTH_EQUATORIAL_RADIUS_KM / np.sqrt(1.0 - EARTH_ECCENTRICITY_SQ * sin_lat**2)

    x_ecef = (N + alt_km) * cos_lat * cos_lon
    y_ecef = (N + alt_km) * cos_lat * sin_lon
    z_ecef = (N * (1.0 - EARTH_ECCENTRICITY_SQ) + alt_km) * sin_lat

    return np.array([x_ecef, y_ecef, z_ecef])


def get_earth_rotation_angle_rad(sim_time_sec):
    """
    Calculates Earth's rotation angle based on simulation time.
    Assumes ECI X-axis aligns with ECEF X-axis (towards Greenwich meridian)
    at sim_time_sec = 0 for this simulation's ECI frame.
    A more rigorous implementation would use GMST based on a precise epoch.
    """
    return (OMEGA_EARTH_RAD_PER_SEC * sim_time_sec) % (2 * np.pi)


def ecef_to_eci(ecef_pos_km, earth_rotation_angle_rad):
    """
    Converts ECEF coordinates to ECI using Earth's rotation angle.
    This transformation rotates the ECEF vector by the Earth's rotation angle
    to align it with the ECI frame (assuming ECI is fixed and ECEF rotates).
    r_ECI = R_z(theta_g) * r_ECEF (if theta_g is Earth's rotation from ECI X to ECEF X)
    """
    angle = earth_rotation_angle_rad  # Earth has rotated by this much
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Rotation matrix for Rz(angle)
    # To get ECI coordinates from ECEF coordinates, when ECEF has rotated by 'angle'
    # relative to a fixed ECI frame:
    # x_eci = x_ecef * cos(angle) - y_ecef * sin(angle)
    # y_eci = x_ecef * sin(angle) + y_ecef * cos(angle)
    # z_eci = z_ecef
    x_eci = ecef_pos_km[0] * cos_a - ecef_pos_km[1] * sin_a
    y_eci = ecef_pos_km[0] * sin_a + ecef_pos_km[1] * cos_a
    z_eci = ecef_pos_km[2]
    return np.array([x_eci, y_eci, z_eci])


def equations_of_motion(t, y_state_vector):
    r_vec = y_state_vector[0:3]
    v_vec = y_state_vector[3:6]
    r_norm = np.linalg.norm(r_vec)
    if r_norm == 0:
        return np.zeros(6)
    a_vec = -MU_EARTH_KM3_S2 * r_vec / (r_norm**3)
    return np.concatenate((v_vec, a_vec))
