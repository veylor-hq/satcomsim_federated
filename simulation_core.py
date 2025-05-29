import asyncio
import json
import time
import numpy as np

# --- Physics & Earth Constants ---
MU_EARTH_KM3_S2 = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
EARTH_EQUATORIAL_RADIUS_KM = 6378.137  # WGS84
EARTH_FLATTENING = 1.0 / 298.257223563  # WGS84
EARTH_ECCENTRICITY_SQ = EARTH_FLATTENING * (2.0 - EARTH_FLATTENING)
SIDEREAL_DAY_SECONDS = 86164.0905  # s
OMEGA_EARTH_RAD_PER_SEC = (2 * np.pi) / SIDEREAL_DAY_SECONDS


# --- Coordinate Transformation Functions ---
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


class SimulationCore:
    def __init__(
        self,
        initial_satellite_configs,
        initial_groundstation_configs,
        control_port=60000,
        vis_port=60001,
        gs_update_port=60002,
    ):

        self.satellite_configs_map = {
            cfg["id"]: cfg for cfg in initial_satellite_configs
        }
        self.satellite_connections = {}  # {sat_id: (reader, writer)}
        self.satellite_states = {
            cfg["id"]: np.array(cfg["initial_state_vector"])
            for cfg in initial_satellite_configs
        }

        self.initial_groundstation_configs = initial_groundstation_configs
        self.groundstation_states = (
            {}
        )  # {gs_id: {"lla": (lat,lon,alt), "ecef": np.array, "eci": np.array}}

        self.simulation_time_sec = 0.0
        self.time_step_sec = 1.0

        self.control_port = control_port
        self.vis_port = vis_port
        self.gs_update_port = gs_update_port

        self.control_server_task = None
        self.visualization_server_task = None
        self.groundstation_update_server_task = None

        self.visualizer_writers = []
        self.groundstation_writers = (
            []
        )  # For clients (groundstation services) connecting to the core

        self.lock = asyncio.Lock()

        # Initialize groundstation ECEF positions (fixed) and initial ECI
        for gs_config in self.initial_groundstation_configs:
            gs_id = gs_config["id"]
            lat = gs_config["lat_deg"]
            lon = gs_config["lon_deg"]
            alt = gs_config["alt_km"]
            ecef_pos = lla_to_ecef(lat, lon, alt)
            self.groundstation_states[gs_id] = {
                "id": gs_id,  # Store id here as well for convenience
                "lla": (lat, lon, alt),
                "ecef": ecef_pos,
                "eci": ecef_to_eci(
                    ecef_pos, get_earth_rotation_angle_rad(self.simulation_time_sec)
                ),  # Initial ECI
            }

    async def _connect_one_satellite(self, sat_id, host, port):
        print(
            f"[Core] Attempting to connect to satellite service {sat_id} at {host}:{port}..."
        )
        try:
            reader, writer = await asyncio.open_connection(host, port)
            print(f"[Core] Successfully established raw connection to {sat_id}.")
            return reader, writer
        except ConnectionRefusedError:
            raise ConnectionRefusedError(
                f"Connection refused by {sat_id} at {host}:{port}"
            )
        except Exception as e:
            raise Exception(
                f"Failed to connect to {sat_id} during _connect_one_satellite: {e}"
            )

    async def connect_to_initial_satellites(self):
        print("[Core] Connecting to initial satellites...")
        connected_count = 0
        for sat_id, config in list(self.satellite_configs_map.items()):
            if sat_id not in self.satellite_connections:
                try:
                    reader, writer = await self._connect_one_satellite(
                        sat_id, config["host"], config["port"]
                    )
                    async with self.lock:
                        self.satellite_connections[sat_id] = (reader, writer)
                    print(
                        f"[Core] Successfully connected to initial satellite {sat_id}."
                    )
                    connected_count += 1
                except Exception as e:
                    print(
                        f"[Core] Failed to connect to initial satellite {sat_id}. Reason: {e}"
                    )
                    async with self.lock:
                        self.satellite_configs_map.pop(sat_id, None)
                        self.satellite_states.pop(sat_id, None)
                        self.satellite_connections.pop(sat_id, None)

        if (
            not self.satellite_configs_map and initial_satellite_configurations
        ):  # Check original list passed to main
            print(
                "[Core] No initial satellites were configured or all failed to connect."
            )
        elif not self.satellite_configs_map and not initial_satellite_configurations:
            print("[Core] No initial satellites were configured.")

        return connected_count > 0 or not initial_satellite_configurations

    async def add_satellite_dynamically(self, config_dict):
        sat_id = config_dict["id"]
        host = config_dict["host"]
        port = config_dict["port"]
        initial_state = np.array(config_dict["initial_state_vector"])
        async with self.lock:
            if sat_id in self.satellite_configs_map:
                print(
                    f"[Core] Satellite {sat_id} already exists. Cannot add dynamically."
                )
                return False
        print(
            f"[Core] Attempting to dynamically add satellite: {sat_id} at {host}:{port}"
        )
        try:
            reader, writer = await self._connect_one_satellite(sat_id, host, port)
            async with self.lock:
                self.satellite_configs_map[sat_id] = config_dict
                self.satellite_connections[sat_id] = (reader, writer)
                self.satellite_states[sat_id] = initial_state
            print(
                f"[Core] Successfully connected and added satellite {sat_id} dynamically."
            )
            return True
        except Exception as e:
            print(
                f"[Core] Failed to connect/add satellite {sat_id} dynamically. Reason: {e}"
            )
            async with (
                self.lock
            ):  # Ensure cleanup if connection failed after config_map was populated (though unlikely with current flow)
                self.satellite_configs_map.pop(sat_id, None)
            return False

    async def handle_control_command(self, reader, writer):
        peer_name = writer.get_extra_info("peername")
        print(f"[CoreControl] Accepted control connection from {peer_name}")
        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    break
                command_str = data.decode().strip()
                print(f"[CoreControl] Received command from {peer_name}: {command_str}")
                response_msg = "ERROR: Unknown command\n"
                try:
                    if command_str.startswith("ADD_SATELLITE "):
                        parts = command_str.split(" ", 1)
                        if len(parts) == 2:
                            new_sat_config = json.loads(parts[1])
                            if not all(
                                k in new_sat_config
                                for k in ["id", "host", "port", "initial_state_vector"]
                            ):
                                response_msg = (
                                    "ERROR: Invalid satellite config format.\n"
                                )
                            else:
                                if await self.add_satellite_dynamically(new_sat_config):
                                    response_msg = f"OK: Satellite {new_sat_config['id']} addition initiated.\n"
                                else:
                                    response_msg = f"ERROR: Failed to add satellite {new_sat_config['id']}.\n"
                        else:
                            response_msg = "ERROR: Malformed ADD_SATELLITE command.\n"
                except json.JSONDecodeError:
                    response_msg = "ERROR: Invalid JSON payload.\n"
                except Exception as e:
                    response_msg = f"ERROR: Internal error: {e}\n"
                writer.write(response_msg.encode())
                await writer.drain()
        except (ConnectionResetError, asyncio.CancelledError):
            pass  # Client disconnected or task cancelled
        except Exception as e:
            print(f"[CoreControl] Error with {peer_name}: {e}")
        finally:
            print(f"[CoreControl] Closing control connection with {peer_name}")
            if writer and not writer.is_closing():
                try:
                    writer.close()
                    await writer.wait_closed()
                except:
                    pass

    async def handle_visualizer_connection(self, reader, writer):
        peer_name = writer.get_extra_info("peername")
        print(f"[CoreVis] Visualizer connected from {peer_name}")
        async with self.lock:
            self.visualizer_writers.append(writer)
        try:
            while True:  # Keep connection alive, data is pushed from sim loop
                if not await reader.read(100):
                    break  # Client disconnected
        except (
            ConnectionResetError,
            asyncio.IncompleteReadError,
            asyncio.CancelledError,
        ):
            pass
        except Exception as e:
            print(f"[CoreVis] Visualizer {peer_name} error: {e}")
        finally:
            print(f"[CoreVis] Visualizer {peer_name} disconnected.")
            async with self.lock:
                if writer in self.visualizer_writers:
                    self.visualizer_writers.remove(writer)
            if writer and not writer.is_closing():
                try:
                    writer.close()
                    await writer.wait_closed()
                except:
                    pass

    async def handle_groundstation_connection(self, reader, writer):
        peer_name = writer.get_extra_info("peername")
        print(f"[CoreGS] Groundstation client connected from {peer_name}")
        # Here, GS client could send its ID for targeted updates, or just receive all GS data
        async with self.lock:
            self.groundstation_writers.append(writer)
        try:
            while True:  # Keep connection alive
                if not await reader.read(100):
                    break
        except (
            ConnectionResetError,
            asyncio.IncompleteReadError,
            asyncio.CancelledError,
        ):
            pass
        except Exception as e:
            print(f"[CoreGS] Groundstation {peer_name} error: {e}")
        finally:
            print(f"[CoreGS] Groundstation client {peer_name} disconnected.")
            async with self.lock:
                if writer in self.groundstation_writers:
                    self.groundstation_writers.remove(writer)
            if writer and not writer.is_closing():
                try:
                    writer.close()
                    await writer.wait_closed()
                except:
                    pass

    def update_groundstation_eci_positions(self):
        earth_rot_angle_rad = get_earth_rotation_angle_rad(self.simulation_time_sec)
        for gs_id in self.groundstation_states:  # Iterate keys, modify dict value
            # ECEF is fixed, so recalculate ECI based on current time
            self.groundstation_states[gs_id]["eci"] = ecef_to_eci(
                self.groundstation_states[gs_id]["ecef"], earth_rot_angle_rad
            )

    def calculate_visibility(
        self, gs_eci_pos_np, sat_eci_pos_np, min_elevation_deg=5.0
    ):
        vec_gs_to_sat_eci_np = sat_eci_pos_np - gs_eci_pos_np
        norm_gs_pos = np.linalg.norm(gs_eci_pos_np)
        if norm_gs_pos < 1e-6:
            return False
        local_up_eci_np = gs_eci_pos_np / norm_gs_pos

        norm_vec_gs_to_sat = np.linalg.norm(vec_gs_to_sat_eci_np)
        if norm_vec_gs_to_sat < 1e-6:
            return True  # Sat at GS

        dot_product = np.dot(vec_gs_to_sat_eci_np, local_up_eci_np)

        # Basic check: if satellite is below the plane passing through Earth's center and normal to GS's position
        # This is a coarse "is it on the same side of Earth" check.
        # A more rigorous check involves line-of-sight not intersecting Earth's body.
        # For simplicity, we use elevation angle based on local up.
        # If dot_product is negative, it means the satellite is in the hemisphere "below" the groundstation's
        # local tangent plane (more or less, assuming spherical Earth for this simple 'up' vector).
        if (
            dot_product <= 0
        ):  # Effectively, elevation <= 0, not accounting for Earth curvature yet for LOS block
            # A true LOS check would be more complex here.
            # This simple check means it's below the mathematical horizon plane at the GS.
            # For sats very far away, this is fine. For close LEO sats, Earth blockage is key.
            # For now, if dot product is < threshold_for_min_elevation, it's not visible.
            # Let's calculate actual elevation:
            pass  # Fall through to arcsin calculation

        sin_elevation = dot_product / norm_vec_gs_to_sat
        sin_elevation = np.clip(sin_elevation, -1.0, 1.0)  # Clamp for arcsin
        elevation_rad = np.arcsin(sin_elevation)
        elevation_deg = np.degrees(elevation_rad)

        # Line of Sight (LOS) check: Does the path intersect Earth?
        # Simplified LOS: if elevation is positive, does it clear the Earth's bulge?
        # For a spherical Earth, positive elevation means LOS is clear.
        # For oblate Earth and low elevation, this can be more complex.
        # We will assume positive elevation means LOS for this simulation level.

        return elevation_deg >= min_elevation_deg

    async def broadcast_to_groundstations(self):
        active_gs_writers = []
        async with self.lock:
            active_gs_writers = list(self.groundstation_writers)
        if not active_gs_writers:
            return

        # Get a consistent snapshot of satellite states for this broadcast
        current_satellite_data_for_gs = {}
        async with self.lock:
            for sat_id, state_vector in self.satellite_states.items():
                if sat_id in self.satellite_connections:  # Only active satellites
                    sat_config = self.satellite_configs_map.get(sat_id)
                    if sat_config:
                        current_satellite_data_for_gs[sat_id] = {
                            "eci_pos": state_vector[0:3].tolist(),
                            "connect_host": sat_config["host"],
                            "connect_port": sat_config["port"],
                        }

        gs_visibility_updates = []
        for gs_id, gs_data_state in self.groundstation_states.items():
            # gs_data_state already has updated "eci" from update_groundstation_eci_positions()
            gs_eci_pos_np = gs_data_state["eci"]
            visible_sats_list = []
            for sat_id, sat_data in current_satellite_data_for_gs.items():
                sat_eci_pos_np = np.array(sat_data["eci_pos"])
                if self.calculate_visibility(
                    gs_eci_pos_np, sat_eci_pos_np
                ):  # Using min_elevation_deg default
                    visible_sats_list.append(
                        {
                            "id": sat_id,
                            # "eci_pos": sat_data["eci_pos"], # GS service might not need this if core tells it
                            "connect_host": sat_data["connect_host"],
                            "connect_port": sat_data["connect_port"],
                        }
                    )
            gs_visibility_updates.append(
                {
                    "id": gs_id,
                    "eci_pos_km": gs_eci_pos_np.tolist(),  # Send GS its own current ECI
                    "visible_sats": visible_sats_list,
                }
            )

        if (
            not gs_visibility_updates
            and self.initial_groundstation_configs
            and self.simulation_time_sec % 10 < self.time_step_sec
        ):
            print(
                "[CoreGS] No groundstation visibility data to send (or no GS configured)."
            )

        message = {
            "type": "GS_VISIBILITY_UPDATE",
            "timestamp_sim_sec": round(self.simulation_time_sec, 2),
            "groundstations_data": gs_visibility_updates,
        }
        message_json = json.dumps(message) + "\n"
        encoded_message = message_json.encode()

        disconnected_writers = []
        for writer in active_gs_writers:
            if writer.is_closing():
                disconnected_writers.append(writer)
                continue
            try:
                writer.write(encoded_message)
                await writer.drain()
            except Exception:
                disconnected_writers.append(writer)

        if disconnected_writers:
            async with self.lock:
                for dw in disconnected_writers:
                    if dw in self.groundstation_writers:
                        self.groundstation_writers.remove(dw)
                    if not dw.is_closing():
                        try:
                            dw.close()
                        except:
                            pass

    async def broadcast_to_visualizers(self):
        active_visualizer_writers = []
        async with self.lock:  # Get a stable list of writers
            active_visualizer_writers = list(self.visualizer_writers)

        if not active_visualizer_writers:
            return

        sat_data_for_vis = []
        async with self.lock:  # Consistent read of satellite states and connections
            for sat_id, state_vector in self.satellite_states.items():
                if (
                    sat_id in self.satellite_connections
                ):  # Only send data for active, connected satellites
                    sat_data_for_vis.append(
                        {
                            "id": sat_id,
                            "pos": state_vector[
                                0:3
                            ].tolist(),  # ECI position [x,y,z] in km
                            "vel": state_vector[
                                3:6
                            ].tolist(),  # ECI velocity [vx,vy,vz] in km/s
                        }
                    )

        # ADD GROUNDSTATION DATA TO THE VIS_UPDATE MESSAGE
        gs_data_for_vis = []
        # update_groundstation_eci_positions() should have been called earlier in the main loop
        for gs_id, gs_state_data in self.groundstation_states.items():
            if "eci" in gs_state_data:  # Ensure ECI position is calculated
                gs_data_for_vis.append(
                    {
                        "id": gs_id,
                        "pos": gs_state_data[
                            "eci"
                        ].tolist(),  # Current ECI position of the groundstation
                    }
                )

        if (
            not sat_data_for_vis
            and not gs_data_for_vis
            and self.simulation_time_sec % 10 < self.time_step_sec
        ):
            print(
                "[CoreVis] No active satellites or groundstations to visualize currently."
            )

        message = {
            "type": "VIS_UPDATE",
            "timestamp_sim_sec": round(self.simulation_time_sec, 2),
            "satellites": sat_data_for_vis,
            "groundstations": gs_data_for_vis,  # NEW: Add groundstation data
        }
        message_json = json.dumps(message) + "\n"  # Newline for easier client parsing
        encoded_message = message_json.encode()
        disconnected_writers = []
        for writer in active_visualizer_writers:  # Similar send logic as above
            if writer.is_closing():
                disconnected_writers.append(writer)
                continue
            try:
                writer.write(encoded_message)
                await writer.drain()
            except Exception:
                disconnected_writers.append(writer)
        if disconnected_writers:  # Same cleanup as above
            async with self.lock:
                for dw in disconnected_writers:
                    if dw in self.visualizer_writers:
                        self.visualizer_writers.remove(dw)
                    if not dw.is_closing():
                        try:
                            dw.close()
                        except:
                            pass

    def equations_of_motion(self, t, y_state_vector):
        r_vec = y_state_vector[0:3]
        v_vec = y_state_vector[3:6]
        r_norm = np.linalg.norm(r_vec)
        if r_norm == 0:
            return np.zeros(6)
        a_vec = -MU_EARTH_KM3_S2 * r_vec / (r_norm**3)
        return np.concatenate((v_vec, a_vec))

    def rk4_step(self, y_state_vector, t, dt):
        k1 = dt * self.equations_of_motion(t, y_state_vector)
        k2 = dt * self.equations_of_motion(t + 0.5 * dt, y_state_vector + 0.5 * k1)
        k3 = dt * self.equations_of_motion(t + 0.5 * dt, y_state_vector + 0.5 * k2)
        k4 = dt * self.equations_of_motion(t + dt, y_state_vector + k3)
        return y_state_vector + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    async def run_simulation_loop(self):
        if not await self.connect_to_initial_satellites():
            if self.satellite_configs_map:
                print(
                    "[Core] Some initial satellites failed to connect. Continuing with available/dynamic ones."
                )

        print("[Core] Starting simulation loop...")
        try:
            while True:
                self.update_groundstation_eci_positions()  # Update GS ECI positions based on Earth rotation

                async with self.lock:
                    active_sat_ids_for_step = list(self.satellite_connections.keys())

                if not active_sat_ids_for_step and self.simulation_time_sec > 0:
                    if self.simulation_time_sec % 30 < self.time_step_sec:
                        print(
                            f"[Core] SimTime: {self.simulation_time_sec:.1f}s. No active satellites. Idling..."
                        )
                elif active_sat_ids_for_step:
                    if (
                        self.simulation_time_sec % 10 < self.time_step_sec
                    ):  # Print less frequently
                        print(
                            f"\n[Core] SimTime: {self.simulation_time_sec:.1f}s. Processing {len(active_sat_ids_for_step)} sats."
                        )

                for sat_id in active_sat_ids_for_step:
                    current_state_vector, writer = None, None
                    async with self.lock:
                        if (
                            sat_id in self.satellite_states
                            and sat_id in self.satellite_connections
                        ):
                            current_state_vector = self.satellite_states[sat_id]
                            _, writer = self.satellite_connections[sat_id]
                        else:
                            continue  # Should not happen if active_sat_ids_for_step is from connections

                    new_state_vector = self.rk4_step(
                        current_state_vector,
                        self.simulation_time_sec,
                        self.time_step_sec,
                    )
                    async with self.lock:
                        self.satellite_states[sat_id] = new_state_vector

                    payload = {
                        "timestamp_sim_sec": round(self.simulation_time_sec, 2),
                        "position_eci_km": new_state_vector[0:3].tolist(),
                        "velocity_eci_km_s": new_state_vector[3:6].tolist(),
                    }
                    message = {"type": "STATE_UPDATE", "payload": payload}
                    message_json = json.dumps(message) + "\n"
                    try:
                        writer.write(message_json.encode())
                        await writer.drain()
                    except (
                        ConnectionResetError,
                        BrokenPipeError,
                        ConnectionAbortedError,
                    ) as e:
                        print(f"[Core] Sat {sat_id} connection lost: {e}. Removing.")
                        async with self.lock:
                            if writer and not writer.is_closing():
                                writer.close()
                            self.satellite_connections.pop(sat_id, None)
                            self.satellite_states.pop(sat_id, None)
                    except Exception as e_send:
                        print(f"[Core] Error sending to sat {sat_id}: {e_send}")

                await self.broadcast_to_groundstations()
                await self.broadcast_to_visualizers()

                self.simulation_time_sec += self.time_step_sec
                await asyncio.sleep(self.time_step_sec)
        except asyncio.CancelledError:
            print("[Core] Simulation loop cancelled.")
        finally:
            print("[Core] Sim loop ended. Closing satellite connections...")
            async with self.lock:
                for sat_id in list(self.satellite_connections.keys()):
                    _, writer = self.satellite_connections.pop(sat_id)
                    if writer and not writer.is_closing():
                        try:
                            writer.close()
                            await writer.wait_closed()
                        except Exception as e:
                            print(f"[Core] Error closing sat {sat_id} conn: {e}")
            print("[Core] All satellite connections closed.")


async def core_main_runner(
    initial_satellite_configs,
    initial_groundstation_configs,
    sim_control_port,
    sim_vis_port,
    sim_gs_update_port,
):
    core = SimulationCore(
        initial_satellite_configs=initial_satellite_configs,
        initial_groundstation_configs=initial_groundstation_configs,
        control_port=sim_control_port,
        vis_port=sim_vis_port,
        gs_update_port=sim_gs_update_port,
    )
    servers = []
    server_tasks = []

    try:
        control_server = await asyncio.start_server(
            core.handle_control_command, "localhost", core.control_port
        )
        servers.append(control_server)
        addr = control_server.sockets[0].getsockname()
        print(f"[Core] Control interface listening on {addr}")
        server_tasks.append(
            asyncio.create_task(control_server.serve_forever(), name="ControlServer")
        )

        vis_server = await asyncio.start_server(
            core.handle_visualizer_connection, "localhost", core.vis_port
        )
        servers.append(vis_server)
        addr = vis_server.sockets[0].getsockname()
        print(f"[Core] Visualization interface listening on {addr}")
        server_tasks.append(
            asyncio.create_task(vis_server.serve_forever(), name="VisServer")
        )

        gs_update_server = await asyncio.start_server(
            core.handle_groundstation_connection, "localhost", core.gs_update_port
        )
        servers.append(gs_update_server)
        addr = gs_update_server.sockets[0].getsockname()
        print(f"[Core] Groundstation Update interface listening on {addr}")
        server_tasks.append(
            asyncio.create_task(gs_update_server.serve_forever(), name="GSUpdateServer")
        )

        simulation_task = asyncio.create_task(
            core.run_simulation_loop(), name="SimulationLoop"
        )
        await simulation_task
    except OSError as e:
        print(
            f"[Core] FATAL: Could not start a server: {e}. Ensure ports are free. Exiting."
        )
    except asyncio.CancelledError:
        print("[Core] Main runner task cancelled.")
    except Exception as e_main_run:
        print(f"[Core] Exception in core_main_runner: {e_main_run}")
    finally:
        print("[Core] Shutting down server tasks and interfaces...")
        for task in server_tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass  # Expected
                except Exception as e_task_cancel:
                    print(
                        f"[Core] Error cancelling server task {task.get_name()}: {e_task_cancel}"
                    )

        for server in servers:
            if server and server.is_serving():
                server.close()
                try:
                    await server.wait_closed()
                except Exception as e_server_close:
                    print(f"[Core] Error closing server: {e_server_close}")
        print("[Core] All server interfaces shut down.")


if __name__ == "__main__":
    initial_satellite_configurations = [
        {
            "id": "SatAlpha_LEO",
            "host": "localhost",
            "port": 65432,
            "initial_state_vector": [6371.0 + 700.0, 0.0, 0.0, 0.0, 7.5, 0.0],
        },
        {
            "id": "SatBeta_MEO",
            "host": "localhost",
            "port": 65433,
            "initial_state_vector": [20000.0, 0.0, 0.0, 0.0, 0.0, 3.87],
        },
    ]
    initial_groundstation_configurations = [
        {"id": "GS_London", "lat_deg": 51.5074, "lon_deg": -0.1278, "alt_km": 0.035},
        {"id": "GS_NewYork", "lat_deg": 40.7128, "lon_deg": -74.0060, "alt_km": 0.010},
        {"id": "GS_Sydney", "lat_deg": -33.8688, "lon_deg": 151.2093, "alt_km": 0.058},
    ]

    CORE_CONTROL_PORT = 60000
    CORE_VIS_PORT = 60001
    CORE_GS_UPDATE_PORT = 60002

    try:
        asyncio.run(
            core_main_runner(
                initial_satellite_configurations,
                initial_groundstation_configurations,
                CORE_CONTROL_PORT,
                CORE_VIS_PORT,
                CORE_GS_UPDATE_PORT,
            )
        )
    except KeyboardInterrupt:
        print("\n[Core] Simulation Core stopped by user (KeyboardInterrupt).")
    except Exception as e_global:
        print(f"[Core] A critical error occurred in the main execution: {e_global}")
    finally:
        print("[Core] Application shutdown complete.")
