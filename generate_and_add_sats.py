# generate_and_add_sats.py
import socket
import json
import random
import numpy as np
import time
import subprocess  # For launching satellite services
import os
import atexit  # For cleanup

# --- Configuration ---
CORE_CONTROL_HOST = "localhost"
CORE_CONTROL_PORT = 60000
NUM_SATELLITES = 30  # You can reduce this for quicker testing initially
BASE_SATELLITE_PORT = 65450
SATELLITE_HOST = "localhost"
PYTHON_EXECUTABLE = "python"  # or "python3" or specific path if needed

# Earth and Orbit Parameters (same as before)
MU_EARTH_KM3_S2 = 398600.4418
EARTH_RADIUS_KM = 6371.0
MIN_ALTITUDE_KM = 700.0
MAX_ALTITUDE_KM = 1500.0  # Reduced max altitude for potentially more LEO satellites
SPEED_VARIATION_FACTOR_MIN = 0.98
SPEED_VARIATION_FACTOR_MAX = 1.02

# --- Process Management ---
launched_satellite_processes = []


def cleanup_launched_processes():
    print("\n[Orchestrator] Cleaning up launched satellite service processes...")
    for i, proc_info in enumerate(launched_satellite_processes):
        proc = proc_info["process"]
        sat_id = proc_info["id"]
        if proc.poll() is None:  # Check if process is still running
            print(
                f"[Orchestrator] Terminating satellite service for {sat_id} (PID: {proc.pid})..."
            )
            proc.terminate()  # Send SIGTERM
            try:
                proc.wait(timeout=5)  # Wait for graceful termination
                print(f"[Orchestrator] {sat_id} terminated.")
            except subprocess.TimeoutExpired:
                print(
                    f"[Orchestrator] {sat_id} did not terminate gracefully, killing..."
                )
                proc.kill()  # Send SIGKILL
                print(f"[Orchestrator] {sat_id} killed.")
        else:
            print(f"[Orchestrator] Satellite service for {sat_id} already terminated.")
    launched_satellite_processes.clear()
    print("[Orchestrator] Cleanup complete.")


# Register cleanup function to be called on script exit
atexit.register(cleanup_launched_processes)


# --- Helper Functions (generate_satellite_configs and send_command_to_core are mostly the same) ---
def generate_satellite_configs(num_sats, base_port):
    configs = []
    print(f"[Orchestrator] Generating {num_sats} satellite configurations...")
    for i in range(num_sats):
        sat_id = f"SatGen_{i:02d}"
        port = base_port + i
        altitude_km = random.uniform(MIN_ALTITUDE_KM, MAX_ALTITUDE_KM)
        r_magnitude_km = EARTH_RADIUS_KM + altitude_km
        u = random.uniform(-1.0, 1.0)
        theta = random.uniform(0, 2 * np.pi)
        x_km = r_magnitude_km * np.sqrt(1 - u**2) * np.cos(theta)
        y_km = r_magnitude_km * np.sqrt(1 - u**2) * np.sin(theta)
        z_km = r_magnitude_km * u
        position_vector_km = np.array([x_km, y_km, z_km])
        v_circular_km_s = np.sqrt(MU_EARTH_KM3_S2 / r_magnitude_km)
        speed_factor = random.uniform(
            SPEED_VARIATION_FACTOR_MIN, SPEED_VARIATION_FACTOR_MAX
        )
        v_magnitude_km_s = v_circular_km_s * speed_factor

        velocity_vector_km_s = np.zeros(3)  # Initialize
        max_attempts = 5
        for attempt in range(max_attempts):
            random_k_vector = np.random.rand(3) - 0.5
            if np.linalg.norm(random_k_vector) < 1e-6:
                continue  # Avoid zero vector
            random_k_vector /= np.linalg.norm(random_k_vector)
            velocity_direction = np.cross(position_vector_km, random_k_vector)
            norm_velocity_direction = np.linalg.norm(velocity_direction)
            if norm_velocity_direction > 1e-3:
                velocity_vector_km_s = (
                    velocity_direction / norm_velocity_direction
                ) * v_magnitude_km_s
                break
        else:
            print(
                f"[Orchestrator] Warning: Could not determine a good perpendicular velocity for {sat_id}. Using a simplified fallback."
            )
            # Fallback logic (same as before)
            if abs(position_vector_km[0]) > 1e-3 or abs(position_vector_km[1]) > 1e-3:
                k_vec_alt = np.array([0, 0, 1.0])
            else:
                k_vec_alt = np.array([1.0, 0, 0])
            velocity_direction = np.cross(position_vector_km, k_vec_alt)
            if np.linalg.norm(velocity_direction) > 1e-3:
                velocity_vector_km_s = (
                    velocity_direction / np.linalg.norm(velocity_direction)
                ) * v_magnitude_km_s
            else:
                print(
                    f"[Orchestrator] Critical Warning: All velocity generation methods failed for {sat_id}. Assigning random direction."
                )
                vel_dir_rand = np.random.rand(3) - 0.5
                velocity_vector_km_s = (
                    vel_dir_rand / np.linalg.norm(vel_dir_rand)
                ) * v_magnitude_km_s

        configs.append(
            {
                "id": sat_id,
                "host": SATELLITE_HOST,
                "port": port,
                "initial_state_vector": [round(v, 3) for v in position_vector_km]
                + [round(v, 3) for v in velocity_vector_km_s],
            }
        )
    print("[Orchestrator] Finished generating configurations.")
    return configs


def send_command_to_core(command_str):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(10.0)
            s.connect((CORE_CONTROL_HOST, CORE_CONTROL_PORT))
            s.sendall(command_str.encode("utf-8"))
            response_bytes = s.recv(1024)
            response = response_bytes.decode("utf-8").strip()
            print(f"[Orchestrator] Core Response: {response}")
            return "OK:" in response
    except socket.timeout:
        print(
            f"[Orchestrator] Error: Timeout connecting/communicating with Core at {CORE_CONTROL_HOST}:{CORE_CONTROL_PORT}."
        )
    except ConnectionRefusedError:
        print(
            f"[Orchestrator] Error: Connection refused by Core at {CORE_CONTROL_HOST}:{CORE_CONTROL_PORT}. Is core running?"
        )
    except Exception as e:
        print(f"[Orchestrator] Error sending command to core: {e}")
    return False


if __name__ == "__main__":
    print(
        "[Orchestrator] Starting script to generate, launch, and add multiple satellites..."
    )

    # Create a directory for satellite logs if it doesn't exist
    log_dir = "satellite_logs"
    os.makedirs(log_dir, exist_ok=True)
    print(f"[Orchestrator] Satellite logs will be saved in '{log_dir}/'")

    generated_satellite_configurations = generate_satellite_configs(
        NUM_SATELLITES, BASE_SATELLITE_PORT
    )

    print(f"\n[Orchestrator] --- Launching {NUM_SATELLITES} Satellite Services ---")

    try:
        for i, config in enumerate(generated_satellite_configurations):
            sat_id = config["id"]
            port = config["port"]
            host = config[
                "host"
            ]  # Should be localhost for satellite_service itself if core is on localhost

            command_list = [
                PYTHON_EXECUTABLE,
                "satellite_service.py",
                "--id",
                sat_id,
                "--host",
                host,  # satellite_service listens on this
                "--port",
                str(port),
            ]

            # Create log files for each satellite's stdout and stderr
            stdout_log_path = os.path.join(log_dir, f"{sat_id}_stdout.log")
            stderr_log_path = os.path.join(log_dir, f"{sat_id}_stderr.log")

            print(
                f"[Orchestrator] ({i+1}/{NUM_SATELLITES}) Launching: {' '.join(command_list)}"
            )
            print(f"              Logs: {stdout_log_path}, {stderr_log_path}")

            with open(stdout_log_path, "wb") as f_out, open(
                stderr_log_path, "wb"
            ) as f_err:
                process = subprocess.Popen(command_list, stdout=f_out, stderr=f_err)

            launched_satellite_processes.append(
                {"id": sat_id, "process": process, "port": port}
            )

            # Give the satellite service a moment to start up
            time.sleep(0.75)  # Increased delay slightly

            # Now send the ADD_SATELLITE command to the core
            add_command_payload = json.dumps(
                {
                    "id": sat_id,
                    "host": SATELLITE_HOST,  # Host the *core* should connect to (usually localhost)
                    "port": port,
                    "initial_state_vector": config["initial_state_vector"],
                }
            )
            full_command_str = f"ADD_SATELLITE {add_command_payload}\n"

            print(
                f"[Orchestrator] Sending ADD_SATELLITE command for {sat_id} to core..."
            )
            send_command_to_core(full_command_str)
            print("-" * 30)

        print(
            f"\n[Orchestrator] All {NUM_SATELLITES} satellite services launched and ADD commands sent."
        )
        print("[Orchestrator] Simulation should be running with these satellites.")
        print(
            "[Orchestrator] Press Ctrl+C to stop this script and terminate launched satellite services."
        )

        # Keep the main script alive while services run, until Ctrl+C
        while True:
            time.sleep(1)
            # Optionally, could check health of subprocesses here if needed

    except KeyboardInterrupt:
        print("\n[Orchestrator] KeyboardInterrupt received. Initiating cleanup...")
    finally:
        cleanup_launched_processes()  # This will also be called by atexit, but calling it here ensures it runs before atexit if Ctrl+C
        print("[Orchestrator] Script finished.")
