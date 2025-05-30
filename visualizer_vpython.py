# visualizer_vpython.py
import threading
import socket
import json
import time
from vpython import (
    scene,
    canvas,
    vector,
    sphere,
    color,
    rate,
    label,
    textures,
    distant_light,
    arrow,
    curve,
    mag,
    norm,
    checkbox,
)
import numpy as np

# --- Configuration ---
CORE_VIS_HOST = "localhost"
CORE_VIS_PORT = 60001
POS_SCALE_FACTOR = 1000.0
R_EARTH_KM = 6371.0
SAT_RADIUS_DISPLAY_UNITS = R_EARTH_KM / POS_SCALE_FACTOR / 30.0
GS_RADIUS_DISPLAY_UNITS = (
    SAT_RADIUS_DISPLAY_UNITS * 0.9
)  # Groundstations slightly smaller
CURVE_TRAIL_RADIUS_DIVISOR = 5

MU_EARTH_KM3_S2 = 398600.4418
SIDEREAL_DAY_SECONDS = 86164.0905
OMEGA_EARTH_RAD_PER_SEC = (2 * np.pi) / SIDEREAL_DAY_SECONDS

# --- Shared Data ---
latest_visualization_data = None
satellite_states_cache = {}
groundstation_visuals_cache = {}  # For GS visuals
data_lock = threading.Lock()
selected_satellite_id = None
show_all_predictions_checkbox_state = False
previous_sim_time_for_earth_rotation = None


# --- Networking Thread
def network_reception_thread_func(host, port):
    global latest_visualization_data
    connection_attempts = 0
    while True:
        client_socket = None
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if connection_attempts < 3 or connection_attempts % 10 == 0:
                print(
                    f"[VisNet] Attempting to connect to Core Visualizer Stream at {host}:{port}..."
                )
            client_socket.connect((host, port))
            print(f"[VisNet] Connected to Simulation Core ({host}:{port}).")
            connection_attempts = 0
            data_buffer = ""
            while True:
                received_data = client_socket.recv(8192)
                if not received_data:
                    print("[VisNet] Connection closed by the server.")
                    break
                data_buffer += received_data.decode("utf-8", errors="ignore")
                while "\n" in data_buffer:
                    message_str, data_buffer = data_buffer.split("\n", 1)
                    if not message_str.strip():
                        continue
                    try:
                        message = json.loads(message_str)
                        if message.get("type") == "VIS_UPDATE":
                            with data_lock:
                                latest_visualization_data = message
                    except json.JSONDecodeError as jde:
                        print(
                            f"[VisNet] JSON Decode Error: {jde} for message part: '{message_str[:100]}...'"
                        )
        except ConnectionRefusedError:
            connection_attempts += 1
            if connection_attempts < 4 or connection_attempts % 10 == 0:
                print(
                    f"[VisNet] Connection refused by {host}:{port}. Retrying in 5s..."
                )
        except (socket.error, BrokenPipeError, ConnectionResetError) as se:
            connection_attempts += 1
            if connection_attempts < 4 or connection_attempts % 10 == 0:
                print(f"[VisNet] Socket error ({type(se).__name__}). Retrying in 5s...")
        except Exception as e_outer:
            connection_attempts += 1
            if connection_attempts < 4 or connection_attempts % 10 == 0:
                print(
                    f"[VisNet] Unexpected network error: {e_outer}. Retrying in 5s..."
                )
        finally:
            if client_socket:
                try:
                    client_socket.close()
                except:
                    pass
        with data_lock:
            latest_visualization_data = None
        time.sleep(5)


# --- Orbit Propagation Functions
def vis_equations_of_motion(t_sec, state_vector_km_kms_np):
    pos_km_np = state_vector_km_kms_np[0:3]
    vel_km_s_np = state_vector_km_kms_np[3:6]
    r_mag_km = np.linalg.norm(pos_km_np)
    if r_mag_km < 1e-6:
        return np.zeros(6)
    accel_km_s2_np = (-MU_EARTH_KM3_S2 / (r_mag_km**3)) * pos_km_np
    return np.concatenate((vel_km_s_np, accel_km_s2_np))


def vis_rk4_step(y_state_np, t_sec, dt_sec, eq_motion_func):
    k1 = dt_sec * eq_motion_func(t_sec, y_state_np)
    k2 = dt_sec * eq_motion_func(t_sec + 0.5 * dt_sec, y_state_np + 0.5 * k1)
    k3 = dt_sec * eq_motion_func(t_sec + 0.5 * dt_sec, y_state_np + 0.5 * k2)
    k4 = dt_sec * eq_motion_func(t_sec + dt_sec, y_state_np + k3)
    return y_state_np + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def predict_orbit(current_pos_km_np, current_vel_km_s_np, duration_sec, time_step_sec):
    points_km_list = []
    current_state_np = np.concatenate(
        (np.array(current_pos_km_np), np.array(current_vel_km_s_np))
    )
    t = 0.0
    num_steps = 0
    if time_step_sec > 1e-6:
        num_steps = int(duration_sec / time_step_sec)
    points_km_list.append(list(current_state_np[0:3]))
    for _ in range(num_steps):
        current_state_np = vis_rk4_step(
            current_state_np, t, time_step_sec, vis_equations_of_motion
        )
        points_km_list.append(list(current_state_np[0:3]))
        t += time_step_sec
    return points_km_list


# --- VPython Main Setup
print("[VisPy] Initializing VPython scene...")
scene = canvas(title="SatComSim Real-Time Visualizer", width=1200, height=900)
scene.forward = vector(0.1, -0.5, -1)
scene.up = vector(0, 0, 1)
scene.ambient = color.gray(0.3)
distant_light(direction=vector(0.5, 0.3, 1), color=color.gray(0.6))
distant_light(direction=vector(-0.5, -0.3, -1), color=color.gray(0.4))
earth_display_radius = R_EARTH_KM / POS_SCALE_FACTOR
earth_sphere = sphere(
    pos=vector(0, 0, 0), radius=earth_display_radius, texture=textures.earth
)

print(f"[VisPy] Applying initial Earth texture correction.")
earth_sphere.rotate(
    angle=np.radians(90),
    axis=vector(1, 0, 0),
    origin=vector(0, 0, 0),
)

earth_sphere.rotate(
    angle=np.radians(90),
    axis=vector(0, 0, 1),
    origin=vector(0, 0, 0),
)

axis_len = earth_display_radius * 1.5
arrow(
    pos=vector(0, 0, 0),
    axis=vector(axis_len, 0, 0),
    shaftwidth=axis_len / 100,
    color=color.red,
)
arrow(
    pos=vector(0, 0, 0),
    axis=vector(0, axis_len, 0),
    shaftwidth=axis_len / 100,
    color=color.green,
)
arrow(
    pos=vector(0, 0, 0),
    axis=vector(0, 0, axis_len),
    shaftwidth=axis_len / 100,
    color=color.blue,
)
label(
    pos=vector(axis_len, 0, 0),
    text="X_ECI",
    xoffset=10,
    color=color.red,
    box=False,
    height=14,
)
label(
    pos=vector(0, axis_len, 0),
    text="Y_ECI",
    xoffset=10,
    color=color.green,
    box=False,
    height=14,
)
label(
    pos=vector(0, 0, axis_len),
    text="Z_ECI",
    xoffset=10,
    color=color.blue,
    box=False,
    height=14,
)

# --- UI Elements (same) ---
sim_time_label_vpy = label(
    text="Sim Time: N/A",
    pixel_pos=True,
    xoffset=20,
    yoffset=scene.height - 30,
    align="left",
    box=False,
    height=16,
)
sat_detail_label_vpy = label(
    text="Click on a satellite to see details.",
    pixel_pos=True,
    xoffset=20,
    yoffset=scene.height - 60,
    align="left",
    box=False,
    height=14,
    visible=True,
    line=False,
)


def toggle_all_predictions_action(checkbox):
    global show_all_predictions_checkbox_state
    show_all_predictions_checkbox_state = checkbox.checked
    with data_lock:
        if not show_all_predictions_checkbox_state:
            for sat_id_key in list(satellite_states_cache.keys()):
                if selected_satellite_id != sat_id_key:
                    sat_entry = satellite_states_cache.get(sat_id_key)
                    if sat_entry and sat_entry.get("pred_orbit"):
                        sat_entry["pred_orbit"].visible = False
    update_predicted_orbits_display()


checkbox_all_preds_vpy = checkbox(
    bind=toggle_all_predictions_action, text="Show All Predicted Orbits", checked=False
)
scene.append_to_caption("\n")


# --- Click Handling
def handle_click(evt):
    global selected_satellite_id
    picked_obj = scene.mouse.pick
    newly_selected_id = None
    if picked_obj:
        with data_lock:
            for sat_id_key, sat_state_val in satellite_states_cache.items():
                if (
                    sat_state_val
                    and "obj" in sat_state_val
                    and picked_obj == sat_state_val["obj"]
                ):
                    newly_selected_id = sat_id_key
                    break
    previous_selected_id = selected_satellite_id
    selected_satellite_id = newly_selected_id
    if (
        previous_selected_id
        and previous_selected_id != selected_satellite_id
        and not show_all_predictions_checkbox_state
    ):
        with data_lock:
            prev_sat_entry = satellite_states_cache.get(previous_selected_id)
            if prev_sat_entry and prev_sat_entry.get("pred_orbit"):
                prev_sat_entry["pred_orbit"].visible = False
    update_satellite_details_display()
    update_predicted_orbits_display()


scene.bind("click", handle_click)


# --- Display Update Functions
def update_satellite_details_display():
    global sat_detail_label_vpy
    display_text = "Click on a satellite to see details."
    show_details_for_selected = False
    if selected_satellite_id:
        with data_lock:
            sat_state = satellite_states_cache.get(selected_satellite_id)
        if (
            sat_state
            and "pos_actual_km" in sat_state
            and "vel_actual_km_s" in sat_state
        ):
            pos_km_vec = sat_state["pos_actual_km"]
            vel_km_s_vec = sat_state["vel_actual_km_s"]
            altitude_km = mag(pos_km_vec) - R_EARTH_KM
            speed_km_s = mag(vel_km_s_vec)
            display_text = (
                f"Selected: {selected_satellite_id}\n"
                f"Pos (km): [{pos_km_vec.x:.1f}, {pos_km_vec.y:.1f}, {pos_km_vec.z:.1f}]\n"
                f"Vel (km/s): [{vel_km_s_vec.x:.2f}, {vel_km_s_vec.y:.2f}, {vel_km_s_vec.z:.2f}]\n"
                f"Altitude: {altitude_km:.1f} km, Speed: {speed_km_s:.2f} km/s"
            )
            show_details_for_selected = True
        elif selected_satellite_id:
            display_text = f"Selected: {selected_satellite_id}\n(Data pending...)"
            show_details_for_selected = True
    sat_detail_label_vpy.text = display_text
    sat_detail_label_vpy.visible = True


def update_predicted_orbits_display():  # Uses user's confirmed prediction params
    with data_lock:
        for sat_id, sat_state in satellite_states_cache.items():
            should_predict = (
                show_all_predictions_checkbox_state or selected_satellite_id == sat_id
            )
            pred_orbit_curve = sat_state.get("pred_orbit")
            vpython_sphere_obj = sat_state.get("obj")
            if not pred_orbit_curve or not vpython_sphere_obj:
                continue
            if (
                should_predict
                and "pos_actual_km" in sat_state
                and "vel_actual_km_s" in sat_state
            ):
                prediction_duration_sec = 90 * 120
                prediction_time_step_sec = 10
                current_pos_vpy = sat_state["pos_actual_km"]
                current_vel_vpy = sat_state["vel_actual_km_s"]
                current_pos_np = np.array(
                    [current_pos_vpy.x, current_pos_vpy.y, current_pos_vpy.z]
                )
                current_vel_np = np.array(
                    [current_vel_vpy.x, current_vel_vpy.y, current_vel_vpy.z]
                )
                predicted_points_km = predict_orbit(
                    current_pos_np,
                    current_vel_np,
                    prediction_duration_sec,
                    prediction_time_step_sec,
                )
                pred_orbit_curve.clear()
                if predicted_points_km:
                    scaled_points = [
                        vector(p[0], p[1], p[2]) / POS_SCALE_FACTOR
                        for p in predicted_points_km
                    ]
                    if scaled_points:
                        scaled_points[0] = vpython_sphere_obj.pos  # Force alignment
                    else:
                        scaled_points = [vpython_sphere_obj.pos]
                    if len(scaled_points) > 1:
                        pred_orbit_curve.append(scaled_points)
                    elif scaled_points:
                        pred_orbit_curve.append(scaled_points[0])
                    pred_orbit_curve.visible = True
                else:
                    pred_orbit_curve.visible = False
            else:
                pred_orbit_curve.visible = False


# --- Start Networking Thread
network_thread = threading.Thread(
    target=network_reception_thread_func,
    args=(CORE_VIS_HOST, CORE_VIS_PORT),
    daemon=True,
)
network_thread.start()
print(
    "[VisPy] VPython scene initialized. Network thread started. Waiting for data from Simulation Core..."
)

# --- VPython Update Loop ---
while True:
    rate(30)
    current_data_from_core = None
    with data_lock:
        if latest_visualization_data:
            current_data_from_core = latest_visualization_data

    if current_data_from_core:
        # global previous_sim_time_for_earth_rotation
        current_sim_time = current_data_from_core.get("timestamp_sim_sec")
        if current_sim_time is not None:
            sim_time_label_vpy.text = f"Sim Time: {current_sim_time:.1f}s"
            if previous_sim_time_for_earth_rotation is None:
                previous_sim_time_for_earth_rotation = current_sim_time
            elif current_sim_time > previous_sim_time_for_earth_rotation:
                dt = current_sim_time - previous_sim_time_for_earth_rotation
                angle = OMEGA_EARTH_RAD_PER_SEC * dt
                if abs(angle) > 1e-9:
                    earth_sphere.rotate(
                        angle=angle, axis=vector(0, 0, 1), origin=vector(0, 0, 0)
                    )
            previous_sim_time_for_earth_rotation = current_sim_time

        active_sat_ids = set()
        active_gs_ids = set()  # To track current items for cleanup
        with (
            data_lock
        ):  # Process all data within one lock if cache modifications happen
            # Satellites
            for sat_info in current_data_from_core.get("satellites", []):
                sat_id = sat_info["id"]
                active_sat_ids.add(sat_id)
                pos_km = sat_info["pos"]
                vel_km = sat_info.get("vel", [0, 0, 0])
                actual_pos = vector(pos_km[0], pos_km[1], pos_km[2])
                actual_vel = vector(vel_km[0], vel_km[1], vel_km[2])
                display_pos = actual_pos / POS_SCALE_FACTOR
                if sat_id not in satellite_states_cache:
                    satellite_states_cache[sat_id] = {
                        "obj": sphere(
                            pos=display_pos,
                            radius=SAT_RADIUS_DISPLAY_UNITS,
                            color=color.orange,
                            make_trail=True,
                            trail_type="curve",
                            trail_color=color.yellow,
                            retain=150,
                            trail_radius=SAT_RADIUS_DISPLAY_UNITS
                            / CURVE_TRAIL_RADIUS_DIVISOR,
                        ),
                        "label": label(
                            pos=display_pos,
                            text=sat_id,
                            xoffset=15,
                            yoffset=10,
                            space=SAT_RADIUS_DISPLAY_UNITS * 0.5,
                            height=12,
                            border=3,
                            font="sans",
                            color=color.white,
                            background=color.black,
                            opacity=0.7,
                        ),
                        "pred_orbit": curve(
                            color=color.cyan,
                            radius=SAT_RADIUS_DISPLAY_UNITS
                            / CURVE_TRAIL_RADIUS_DIVISOR,
                            visible=False,
                        ),
                    }
                entry = satellite_states_cache[sat_id]
                entry["obj"].pos = display_pos
                entry["label"].pos = display_pos
                entry["pos_actual_km"] = actual_pos
                entry["vel_actual_km_s"] = actual_vel

            # Groundstations
            for gs_info in current_data_from_core.get("groundstations", []):
                gs_id = gs_info["id"]
                active_gs_ids.add(gs_id)
                gs_pos_km = gs_info["pos"]
                gs_display_pos = (
                    vector(gs_pos_km[0], gs_pos_km[1], gs_pos_km[2]) / POS_SCALE_FACTOR
                )
                if gs_id not in groundstation_visuals_cache:
                    groundstation_visuals_cache[gs_id] = {
                        "obj": sphere(
                            pos=gs_display_pos,
                            radius=GS_RADIUS_DISPLAY_UNITS,
                            color=color.green,
                        ),
                        "label": label(
                            pos=gs_display_pos,
                            text=gs_id,
                            xoffset=10,
                            yoffset=-15,
                            height=10,
                            border=2,
                            font="sans",
                            color=color.white,
                            background=color.black,
                            opacity=0.7,
                        ),
                    }
                else:
                    gs_entry = groundstation_visuals_cache[gs_id]
                    gs_entry["obj"].pos = gs_display_pos
                    gs_entry["label"].pos = gs_display_pos

            # Cleanup old satellites
            for sat_id_remove in list(satellite_states_cache.keys()):
                if sat_id_remove not in active_sat_ids:
                    entry = satellite_states_cache.pop(sat_id_remove)
                    entry["obj"].visible = False
                    entry["label"].visible = False
                    entry["pred_orbit"].visible = False
                    if selected_satellite_id == sat_id_remove:
                        selected_satellite_id = None
            # Cleanup old groundstations
            for gs_id_remove in list(groundstation_visuals_cache.keys()):
                if gs_id_remove not in active_gs_ids:
                    entry = groundstation_visuals_cache.pop(gs_id_remove)
                    entry["obj"].visible = False
                    entry["label"].visible = False

        update_satellite_details_display()
        update_predicted_orbits_display()
    else:
        current_label_text = sim_time_label_vpy.text
        waiting_text = "(Waiting for new data...)"
        if not (waiting_text in current_label_text or "N/A" in current_label_text):
            if previous_sim_time_for_earth_rotation is not None:
                sim_time_label_vpy.text = f"Sim Time: {previous_sim_time_for_earth_rotation:.1f}s {waiting_text}"
            else:
                sim_time_label_vpy.text = f"Sim Time: N/A {waiting_text}"
