# launch_london_test_sat.py
import socket
import json
import numpy as np
import time
import subprocess
import os
import atexit

# --- Configuration ---
CORE_CONTROL_HOST = "localhost"
CORE_CONTROL_PORT = 60000
PYTHON_EXECUTABLE = "python"  # or "python3" or specific path

SATELLITE_ID = "SatOverLondon"
SATELLITE_HOST = "localhost"  # Host for the satellite_service.py itself
SATELLITE_PORT = 65440  # Dedicated port for this test satellite service

# Earth and Orbit Parameters
MU_EARTH_KM3_S2 = 398600.4418
EARTH_RADIUS_KM = 6371.0
ALTITUDE_KM = 700.0

# --- Process Management ---
launched_satellite_process_info = (
    None  # To store info about the single launched process
)

groundstation_visuals_cache = {}


def cleanup_launched_process():
    global launched_satellite_process_info
    if launched_satellite_process_info:
        proc = launched_satellite_process_info["process"]
        sat_id = launched_satellite_process_info["id"]
        if proc.poll() is None:  # Check if process is still running
            print(
                f"[Launcher] Terminating satellite service for {sat_id} (PID: {proc.pid})..."
            )
            proc.terminate()
            try:
                proc.wait(timeout=5)
                print(f"[Launcher] {sat_id} terminated.")
            except subprocess.TimeoutExpired:
                print(f"[Launcher] {sat_id} did not terminate gracefully, killing...")
                proc.kill()
                print(f"[Launcher] {sat_id} killed.")
        else:
            print(f"[Launcher] Satellite service for {sat_id} already terminated.")
        launched_satellite_process_info = None
    print("[Launcher] Cleanup complete.")


atexit.register(cleanup_launched_process)


def calculate_initial_state():
    """Calculates initial ECI state for a polar orbit passing over 0-deg longitude."""
    r_magnitude_km = EARTH_RADIUS_KM + ALTITUDE_KM
    v_magnitude_km_s = np.sqrt(MU_EARTH_KM3_S2 / r_magnitude_km)

    # Initial Position: On ECI X-axis (equator, 0-deg longitude at t=0 in our simplified ECI/ECEF alignment)
    position_vector_km = np.array([r_magnitude_km, 0.0, 0.0])

    # Initial Velocity: Moving North along ECI Z-axis (for a polar orbit starting from X-axis on equator)
    velocity_vector_km_s = np.array([0.0, 0.0, v_magnitude_km_s])

    # For a more conventional polar orbit where it crosses equator on X moving along +Y:
    # position_vector_km = np.array([r_magnitude_km, 0.0, 0.0])
    # velocity_vector_km_s = np.array([0.0, v_magnitude_km_s, 0.0])
    # The Z-velocity makes it go "up and over the pole" from X-axis start.

    return {
        "id": SATELLITE_ID,
        "host": SATELLITE_HOST,  # Host for the core to connect to this satellite service
        "port": SATELLITE_PORT,
        "initial_state_vector": [
            round(position_vector_km[0], 3),
            round(position_vector_km[1], 3),
            round(position_vector_km[2], 3),
            round(velocity_vector_km_s[0], 3),
            round(velocity_vector_km_s[1], 3),
            round(velocity_vector_km_s[2], 3),
        ],
    }


def send_command_to_core(command_str):
    """Sends a command string to the Simulation Core's control port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(10.0)
            s.connect((CORE_CONTROL_HOST, CORE_CONTROL_PORT))
            s.sendall(command_str.encode("utf-8"))
            response_bytes = s.recv(1024)
            response = response_bytes.decode("utf-8").strip()
            print(f"[Launcher] Core Response: {response}")
            return "OK:" in response
    except socket.timeout:
        print(
            f"[Launcher] Error: Timeout connecting/communicating with Core at {CORE_CONTROL_HOST}:{CORE_CONTROL_PORT}."
        )
    except ConnectionRefusedError:
        print(
            f"[Launcher] Error: Connection refused by Core at {CORE_CONTROL_HOST}:{CORE_CONTROL_PORT}. Is core running?"
        )
    except Exception as e:
        print(f"[Launcher] Error sending command to core: {e}")
    return False


if __name__ == "__main__":
    print(
        "[Launcher] Starting script to launch and add a test satellite over London..."
    )

    log_dir = "satellite_logs"
    os.makedirs(log_dir, exist_ok=True)

    sat_config = calculate_initial_state()
    sat_id = sat_config["id"]
    listen_port_for_sat_service = sat_config[
        "port"
    ]  # Port the satellite service will listen on

    command_list_for_sat_service = [
        PYTHON_EXECUTABLE,
        "satellite_service.py",
        "--id",
        sat_id,
        "--host",
        SATELLITE_HOST,  # satellite_service.py listens on this host
        "--port",
        str(listen_port_for_sat_service),
    ]

    stdout_log_path = os.path.join(log_dir, f"{sat_id}_stdout.log")
    stderr_log_path = os.path.join(log_dir, f"{sat_id}_stderr.log")

    print(
        f"[Launcher] Launching satellite service for {sat_id} on port {listen_port_for_sat_service}..."
    )
    print(f"          Command: {' '.join(command_list_for_sat_service)}")
    print(f"          Logs: {stdout_log_path}, {stderr_log_path}")

    try:
        with open(stdout_log_path, "wb") as f_out, open(stderr_log_path, "wb") as f_err:
            process = subprocess.Popen(
                command_list_for_sat_service, stdout=f_out, stderr=f_err
            )

        launched_satellite_process_info = {
            "id": sat_id,
            "process": process,
            "port": listen_port_for_sat_service,
        }
        print(
            f"[Launcher] Satellite service for {sat_id} launched (PID: {process.pid}). Giving it a moment to start..."
        )

        time.sleep(1.5)  # Increased delay to ensure satellite service is ready

        add_command_payload = json.dumps(
            {
                "id": sat_id,
                "host": SATELLITE_HOST,  # Host the *core* should use to connect to the satellite service
                "port": listen_port_for_sat_service,
                "initial_state_vector": sat_config["initial_state_vector"],
            }
        )
        full_command_str_to_core = f"ADD_SATELLITE {add_command_payload}\n"

        print(f"[Launcher] Sending ADD_SATELLITE command for {sat_id} to core...")
        if send_command_to_core(full_command_str_to_core):
            print(
                f"[Launcher] Satellite {sat_id} successfully requested to be added to core."
            )
        else:
            print(
                f"[Launcher] Failed to add satellite {sat_id} to core. Terminating launched service."
            )
            cleanup_launched_process()  # Terminate if core rejected or couldn't be reached
            # sys.exit(1)

        print(f"\n[Launcher] Test satellite {sat_id} should be operational.")
        print(f"[Launcher] Groundstation GS_London (if running) should see it pass.")
        print(
            "[Launcher] Press Ctrl+C to stop this script and terminate the launched satellite service."
        )

        while True:
            time.sleep(1)  # Keep script alive

    except KeyboardInterrupt:
        print("\n[Launcher] KeyboardInterrupt received. Initiating cleanup...")
    except Exception as e:
        print(f"[Launcher] An error occurred: {e}")
    finally:
        cleanup_launched_process()  # Ensures cleanup on exit
        print("[Launcher] Script finished.")
