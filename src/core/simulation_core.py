# simulation_core.py (or simulation_core/core_logic.py)
import asyncio
import json
import numpy as np
import time  # Used in run_simulation_loop for timing

# Using the import structure you provided
from src.shared.transformations import (
    equations_of_motion as eom_func,  # Alias for clarity
    lla_to_ecef,
    ecef_to_eci,
    get_earth_rotation_angle_rad,
)

# MU_EARTH_KM3_S2 is assumed to be accessible by eom_func within its module.
# Earth parameters like EARTH_EQUATORIAL_RADIUS_KM are used by lla_to_ecef
# and are assumed to be defined/accessible within src.shared.transformations.


class SimulationCore:
    def __init__(
        self,
        # No initial_satellite_configs
        # No initial_groundstation_configs
        control_port=60000,
        vis_port=60001,
        gs_update_port=60002,
        time_step_sec=1.0,
    ):
        # Satellites are added dynamically via ADD_SATELLITE command
        self.satellite_configs_map = {}
        self.satellite_connections = {}
        self.satellite_states = {}

        # Groundstations are added dynamically via ADD_GROUNDSTATION command
        self.groundstation_states = {}

        self.simulation_time_sec = 0.0
        self.time_step_sec = time_step_sec

        self.control_port = control_port
        self.vis_port = vis_port
        self.gs_update_port = gs_update_port

        self.control_server_task = None
        self.visualization_server_task = None
        self.groundstation_update_server_task = None

        self.visualizer_writers = []
        self.groundstation_writers = []

        self.lock = asyncio.Lock()

        print(
            "[Core] SimulationCore initialized. Add satellites and groundstations dynamically."
        )

    async def _connect_one_satellite(self, sat_id, host, port):
        print(
            f"[Core] Attempting to connect to satellite service {sat_id} at {host}:{port}..."
        )
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=5.0
            )
            print(f"[Core] Successfully established raw connection to {sat_id}.")
            return reader, writer
        except ConnectionRefusedError:
            raise ConnectionRefusedError(
                f"Connection refused by {sat_id} at {host}:{port}"
            )
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"Timeout connecting to {sat_id} at {host}:{port}"
            )
        except Exception as e:
            raise Exception(
                f"Failed to connect to {sat_id} during _connect_one_satellite: {e}"
            )

    async def add_satellite_dynamically(self, config_dict):
        sat_id = config_dict.get("id")
        host = config_dict.get("host")
        port = config_dict.get("port")
        initial_state_list = config_dict.get("initial_state_vector")

        if not all([sat_id, host, port is not None, initial_state_list is not None]):
            print(
                f"[Core] ERROR: ADD_SATELLITE missing required fields (id, host, port, initial_state_vector) for {sat_id or 'UnknownID'}."
            )
            return False

        if not (
            isinstance(initial_state_list, list)
            and len(initial_state_list) == 6
            and all(isinstance(n, (int, float)) for n in initial_state_list)
        ):
            print(
                f"[Core] ERROR: Invalid initial_state_vector for satellite {sat_id}. Must be list of 6 numbers."
            )
            return False
        initial_state = np.array(initial_state_list)

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
            return False

    async def add_groundstation_dynamically(self, gs_config_dict):
        async with self.lock:
            gs_id = gs_config_dict.get("id")
            if not gs_id:
                print("[Core] ERROR: Groundstation config missing 'id'.")
                return False
            if gs_id in self.groundstation_states:
                print(f"[Core] Groundstation {gs_id} already exists.")
                return False

            lat_deg = gs_config_dict.get("lat_deg")
            lon_deg = gs_config_dict.get("lon_deg")
            alt_km = gs_config_dict.get("alt_km", 0.0)
            min_elevation_deg = gs_config_dict.get("min_elevation_deg", 5.0)

            if lat_deg is None or lon_deg is None:
                print(
                    f"[Core] ERROR: Groundstation {gs_id} missing 'lat_deg' or 'lon_deg'."
                )
                return False
            if not all(
                isinstance(val, (int, float))
                for val in [lat_deg, lon_deg, alt_km, min_elevation_deg]
            ):
                print(f"[Core] ERROR: LLA/MinElev for GS {gs_id} must be numbers.")
                return False

            print(
                f"[Core] Adding GS '{gs_id}' LLA:({lat_deg:.2f}, {lon_deg:.2f}, {alt_km:.2f}), MinEl:{min_elevation_deg:.1f}"
            )
            ecef_pos = lla_to_ecef(lat_deg, lon_deg, alt_km)
            gs_lat_r, gs_lon_r = np.radians(lat_deg), np.radians(lon_deg)
            ecef_norm = np.array(
                [
                    np.cos(gs_lat_r) * np.cos(gs_lon_r),
                    np.cos(gs_lat_r) * np.sin(gs_lon_r),
                    np.sin(gs_lat_r),
                ]
            )
            rot_angle = get_earth_rotation_angle_rad(self.simulation_time_sec)

            self.groundstation_states[gs_id] = {
                "id": gs_id,
                "lla": (lat_deg, lon_deg, alt_km),
                "ecef": ecef_pos,
                "ecef_normal": ecef_norm,
                "eci": ecef_to_eci(ecef_pos, rot_angle),
                "eci_normal": ecef_to_eci(ecef_norm, rot_angle),
                "min_elevation_deg": min_elevation_deg,
            }
            print(f"[Core] Groundstation {gs_id} added successfully.")
            return True

    async def handle_control_command(self, reader, writer):
        peer_name = writer.get_extra_info("peername", "UnknownPeer")
        print(f"[CoreControl] Accepted control connection from {peer_name}")
        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    break
                command_str = data.decode("utf-8").strip()
                print(f"[CoreControl] Received from {peer_name}: {command_str}")
                response_msg = "ERROR: Unknown command or malformed payload.\n"
                try:
                    parts = command_str.split(" ", 1)
                    cmd_type = parts[0].upper()
                    payload_str = parts[1] if len(parts) > 1 else "{}"
                    config_data = json.loads(payload_str)

                    if cmd_type == "ADD_SATELLITE":
                        if await self.add_satellite_dynamically(config_data):
                            response_msg = f"OK: Satellite {config_data.get('id','Unknown')} addition initiated.\n"
                        else:
                            response_msg = f"ERROR: Failed to add satellite {config_data.get('id','Unknown')}.\n"
                    elif cmd_type == "ADD_GROUNDSTATION":
                        if await self.add_groundstation_dynamically(config_data):
                            response_msg = f"OK: Groundstation {config_data.get('id','Unknown')} added.\n"
                        else:
                            response_msg = f"ERROR: Failed to add groundstation {config_data.get('id','Unknown')}.\n"
                except json.JSONDecodeError:
                    response_msg = "ERROR: Invalid JSON payload.\n"
                except Exception as e_cmd:
                    response_msg = f"ERROR: Processing command: {e_cmd}\n"
                    print(f"[CoreControl] Ex: {e_cmd}")
                writer.write(response_msg.encode("utf-8"))
                await writer.drain()
        except (
            ConnectionResetError,
            asyncio.CancelledError,
            asyncio.IncompleteReadError,
        ):
            pass
        except Exception as e_conn:
            print(f"[CoreControl] Connection error with {peer_name}: {e_conn}")
        finally:
            print(f"[CoreControl] Closing control connection with {peer_name}")
            if writer and not writer.is_closing():
                try:
                    writer.close()
                    await writer.wait_closed()
                except:
                    pass

    async def _handle_client_connection_keepalive(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        client_list: list,
        client_type: str,
    ):
        """Generic handler for visualizer and groundstation service clients (who only listen)."""
        peer_name = writer.get_extra_info("peername", "UnknownPeer")
        print(f"[Core-{client_type}] Client connected from {peer_name}")
        async with self.lock:
            client_list.append(writer)
        try:
            while True:  # Keep connection alive, data is pushed from sim loop
                if not await reader.read(1024):
                    break  # Client disconnected
        except (
            ConnectionResetError,
            asyncio.IncompleteReadError,
            asyncio.CancelledError,
        ):
            pass
        except Exception as e:
            print(f"[Core-{client_type}] Client {peer_name} error: {e}")
        finally:
            print(f"[Core-{client_type}] Client {peer_name} disconnected.")
            async with self.lock:
                if writer in client_list:
                    client_list.remove(writer)
            if writer and not writer.is_closing():
                try:
                    writer.close()
                    await writer.wait_closed()
                except:
                    pass

    async def handle_visualizer_connection(self, reader, writer):
        await self._handle_client_connection_keepalive(
            reader, writer, self.visualizer_writers, "Vis"
        )

    async def handle_groundstation_connection(self, reader, writer):
        await self._handle_client_connection_keepalive(
            reader, writer, self.groundstation_writers, "GS"
        )

    def update_groundstation_eci_positions(self):
        earth_rot_angle_rad = get_earth_rotation_angle_rad(self.simulation_time_sec)
        for gs_id in list(
            self.groundstation_states.keys()
        ):  # Iterate over keys if dict might change (not here)
            gs_state = self.groundstation_states[gs_id]
            gs_state["eci"] = ecef_to_eci(gs_state["ecef"], earth_rot_angle_rad)
            if "ecef_normal" in gs_state:
                gs_state["eci_normal"] = ecef_to_eci(
                    gs_state["ecef_normal"], earth_rot_angle_rad
                )
            else:  # Should have been initialized by add_groundstation_dynamically
                print(
                    f"[Core] CRITICAL WARNING: ECEF normal vector missing for GS {gs_id}. Using geocentric."
                )
                norm_gs_eci_pos = np.linalg.norm(gs_state["eci"])
                gs_state["eci_normal"] = (
                    gs_state["eci"] / norm_gs_eci_pos
                    if norm_gs_eci_pos > 1e-6
                    else np.array([0, 0, 1])
                )

    # Using the calculate_visibility signature from your snippet (no logging IDs)
    def calculate_visibility(
        self, gs_eci_pos_np, gs_local_up_eci_np, sat_eci_pos_np, min_elevation_deg=5.0
    ):
        vec_gs_to_sat_eci_np = sat_eci_pos_np - gs_eci_pos_np
        norm_vec_gs_to_sat = np.linalg.norm(vec_gs_to_sat_eci_np)
        if norm_vec_gs_to_sat < 1e-3:
            return True
        dot_product = np.dot(vec_gs_to_sat_eci_np, gs_local_up_eci_np)

        required_min_projection = 0.0
        if norm_vec_gs_to_sat > 1e-6:  # Avoid issues if sat is at GS
            required_min_projection = norm_vec_gs_to_sat * np.sin(
                np.radians(min_elevation_deg - 0.001)
            )  # Tolerance

        if dot_product < required_min_projection:  # Simplified direct check
            return False

        sin_elevation = (
            dot_product / norm_vec_gs_to_sat if norm_vec_gs_to_sat > 1e-6 else 0.0
        )
        sin_elevation = np.clip(sin_elevation, -1.0, 1.0)
        elevation_rad = np.arcsin(sin_elevation)
        elevation_deg = np.degrees(elevation_rad)
        return elevation_deg >= min_elevation_deg

    async def _broadcast_to_client_list(
        self, client_writers_list: list, encoded_message: bytes, list_name_for_log: str
    ):
        """Helper function to broadcast a message to a list of writers."""
        disconnected_writers = []
        # Iterate over a copy of the list if modifications can happen during iteration
        # However, the lock around client_list modification is in the connection handlers.
        # For sending, it's safer to iterate a copy obtained under lock.
        current_writers_snapshot = []
        async with self.lock:
            current_writers_snapshot = list(client_writers_list)

        for writer in current_writers_snapshot:
            if writer.is_closing():
                disconnected_writers.append(writer)
                continue
            try:
                writer.write(encoded_message)
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError) as e:
                print(
                    f"[Core-{list_name_for_log}] Client connection lost during broadcast: {e}."
                )
                disconnected_writers.append(writer)
            except Exception as e_send:
                print(
                    f"[Core-{list_name_for_log}] Send Error during broadcast: {e_send}"
                )
                disconnected_writers.append(writer)

        if disconnected_writers:
            async with self.lock:  # Lock to modify the original client_list
                for dw in disconnected_writers:
                    if (
                        dw in client_writers_list
                    ):  # Check again as it might have been removed
                        client_writers_list.remove(dw)
                    if not dw.is_closing():
                        try:
                            dw.close()  # No await here, just initiate close
                        except:
                            pass

    async def broadcast_to_groundstations(self):
        if not self.groundstation_writers and not self.groundstation_states:
            return  # No one to send to / no data

        current_satellite_data = {}
        async with self.lock:  # Access satellite states under lock
            for sat_id, state_vec in self.satellite_states.items():
                if sat_id in self.satellite_connections:  # Only active satellites
                    cfg = self.satellite_configs_map.get(sat_id)
                    if cfg:
                        current_satellite_data[sat_id] = {
                            "eci_pos_np": state_vec[0:3],
                            "connect_host": cfg["host"],
                            "connect_port": cfg["port"],
                        }

        gs_payload_list = []
        # Access self.groundstation_states safely, copy for iteration if modified elsewhere
        # For now, assuming it's stable during this calculation phase after update_groundstation_eci_positions
        for gs_id, gs_state in self.groundstation_states.items():
            gs_eci = gs_state.get("eci")
            gs_up = gs_state.get("eci_normal")
            min_el = gs_state.get("min_elevation_deg", 5.0)

            if gs_eci is None or gs_up is None:
                continue
            vis_sats = []
            for sat_id, sat_d in current_satellite_data.items():
                if self.calculate_visibility(
                    gs_eci, gs_up, sat_d["eci_pos_np"], min_elevation_deg=min_el
                ):
                    vis_sats.append(
                        {
                            "id": sat_id,
                            "connect_host": sat_d["connect_host"],
                            "connect_port": sat_d["connect_port"],
                        }
                    )
            gs_payload_list.append(
                {"id": gs_id, "eci_pos_km": gs_eci.tolist(), "visible_sats": vis_sats}
            )

        msg = {
            "type": "GS_SIM_UPDATE",
            "timestamp_sim_sec": round(self.simulation_time_sec, 2),
            "groundstations_data": gs_payload_list,
        }
        enc_msg = (json.dumps(msg) + "\n").encode("utf-8")
        await self._broadcast_to_client_list(self.groundstation_writers, enc_msg, "GS")

    async def broadcast_to_visualizers(self):
        if not self.visualizer_writers:
            return

        sat_data = []
        gs_data_vis = []
        async with self.lock:  # Access states under lock
            for sat_id, state_vec in self.satellite_states.items():
                if sat_id in self.satellite_connections:
                    sat_data.append(
                        {
                            "id": sat_id,
                            "pos": state_vec[0:3].tolist(),
                            "vel": state_vec[3:6].tolist(),
                        }
                    )
            for gs_id, gs_state_data in self.groundstation_states.items():
                if "eci" in gs_state_data:
                    gs_data_vis.append(
                        {"id": gs_id, "pos": gs_state_data["eci"].tolist()}
                    )

        msg = {
            "type": "VIS_UPDATE",
            "timestamp_sim_sec": round(self.simulation_time_sec, 2),
            "satellites": sat_data,
            "groundstations": gs_data_vis,
        }
        enc_msg = (json.dumps(msg) + "\n").encode("utf-8")
        await self._broadcast_to_client_list(self.visualizer_writers, enc_msg, "Vis")

    def rk4_step(self, y_state_vector, t, dt):  # This is now the internal RK4 method
        """Performs RK4 step using the imported equations_of_motion (eom_func)."""
        # eom_func is already imported from src.shared.transformations
        k1 = dt * eom_func(t, y_state_vector)
        k2 = dt * eom_func(t + 0.5 * dt, y_state_vector + 0.5 * k1)
        k3 = dt * eom_func(t + 0.5 * dt, y_state_vector + 0.5 * k2)
        k4 = dt * eom_func(t + dt, y_state_vector + k3)
        return y_state_vector + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    async def run_simulation_loop(self):
        print(
            "[Core] Starting simulation loop. Add satellites and groundstations via control port."
        )
        try:
            while True:
                loop_start_time_wall_clock = time.perf_counter()
                self.update_groundstation_eci_positions()
                active_sat_ids_for_step = []
                async with self.lock:
                    active_sat_ids_for_step = list(self.satellite_connections.keys())

                if not active_sat_ids_for_step and self.simulation_time_sec > 0:
                    if self.simulation_time_sec % 30 < self.time_step_sec:
                        print(
                            f"[Core] SimTime: {self.simulation_time_sec:.1f}s. No active satellites. Idling..."
                        )
                elif active_sat_ids_for_step:
                    if self.simulation_time_sec % 10 < self.time_step_sec:
                        print(
                            f"\n[Core] SimTime: {self.simulation_time_sec:.1f}s. Processing {len(active_sat_ids_for_step)} sats."
                        )

                for sat_id in active_sat_ids_for_step:  # Iterate copy of keys
                    current_state_vector, writer = None, None
                    async with self.lock:  # Get current state and writer under lock
                        if (
                            sat_id in self.satellite_states
                            and sat_id in self.satellite_connections
                        ):
                            current_state_vector = self.satellite_states[sat_id]
                            _, writer = self.satellite_connections[sat_id]
                        else:
                            continue  # Sat might have been removed

                    new_state_vector = self.rk4_step(
                        current_state_vector,
                        self.simulation_time_sec,
                        self.time_step_sec,
                    )

                    async with self.lock:
                        if sat_id in self.satellite_states:
                            self.satellite_states[sat_id] = new_state_vector
                        else:
                            continue  # Sat removed during propagation

                    payload = {
                        "timestamp_sim_sec": round(self.simulation_time_sec, 2),
                        "position_eci_km": new_state_vector[0:3].tolist(),
                        "velocity_eci_km_s": new_state_vector[3:6].tolist(),
                    }
                    message = {"type": "STATE_UPDATE", "payload": payload}
                    message_json = json.dumps(message) + "\n"
                    try:
                        if writer and not writer.is_closing():
                            writer.write(message_json.encode("utf-8"))
                            await writer.drain()
                        else:
                            raise ConnectionError(
                                "Writer is closed or None for satellite."
                            )
                    except (
                        ConnectionResetError,
                        BrokenPipeError,
                        ConnectionAbortedError,
                        ConnectionError,
                    ) as e:
                        print(f"[Core] Sat {sat_id} connection lost: {e}. Removing.")
                        async with self.lock:
                            if writer and not writer.is_closing():
                                writer.close()  # Try to close
                            self.satellite_connections.pop(sat_id, None)
                            self.satellite_states.pop(sat_id, None)
                            self.satellite_configs_map.pop(
                                sat_id, None
                            )  # Also remove from configs map
                    except Exception as e_send:
                        print(f"[Core] Error sending state to sat {sat_id}: {e_send}")

                await self.broadcast_to_groundstations()
                await self.broadcast_to_visualizers()

                self.simulation_time_sec += self.time_step_sec
                loop_end_time_wall_clock = time.perf_counter()
                loop_duration = loop_end_time_wall_clock - loop_start_time_wall_clock
                sleep_duration = self.time_step_sec - loop_duration
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)

        except asyncio.CancelledError:
            print("[Core] Simulation loop cancelled.")
        finally:
            print(
                "[Core] Simulation loop ended. Closing all active satellite connections..."
            )
            async with self.lock:
                for sat_id in list(self.satellite_connections.keys()):
                    _, writer = self.satellite_connections.pop(
                        sat_id, (None, None)
                    )  # Pop and get writer
                    if writer and not writer.is_closing():
                        try:
                            writer.close()
                            await writer.wait_closed()
                        except Exception as e_close:
                            print(
                                f"[Core] Error closing satellite {sat_id} connection: {e_close}"
                            )
            print(
                "[Core] All satellite connections from simulation loop cleanup have been closed."
            )
