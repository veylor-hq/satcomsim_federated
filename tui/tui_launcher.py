import asyncio
import json
import os
import socket
import sys
import time
import random
import numpy as np

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Static, Log, Input, Button, Label
from textual.binding import Binding
from textual.reactive import reactive

from src.shared.constants import MU_EARTH_KM3_S2, EARTH_RADIUS_KM

# --- Configuration ---
PYTHON_EXECUTABLE = sys.executable
LOG_DIR_TUI = "simulation_logs_tui"  # For any direct file logging if Popen was used (not primary here)

CORE_SCRIPT = "simulation_core.py"
VISUALIZER_SCRIPT = "scripts/visualizer_vpython.py"
SATELLITE_SCRIPT = "scripts/launch_satellite.py"
GROUNDSTATION_SCRIPT = "scripts/launch_groundstation.py"

CORE_CONTROL_HOST = "localhost"
CORE_CONTROL_PORT = 60000
# Core uses its defaults for VIS (60001) and GS_UPDATE (60002)

NUM_SATELLITES_TUI = 45
SATELLITE_SERVICE_BASE_PORT = 65510  # Start port for 10 sats: 65510-65519
SATELLITE_HOST_FOR_SERVICE = "localhost"  # Host satellite_service.py listens on
SATELLITE_HOST_FOR_CORE = "localhost"  # Host core uses to connect to sat_service

GS_ID_TUI = (
    "GS_London"  # Must match a GS_ID defined in core's initial_groundstation_configs
)
GS_USER_CMD_PORT_TUI = 7000
GS_CORE_HOST_TUI = "localhost"  # For GS service to connect to core's update stream
GS_CORE_PORT_TUI = 60002

MIN_ALTITUDE_KM = 700.0
MAX_ALTITUDE_KM = 1500.0
SPEED_VARIATION_FACTOR_MIN = 0.98
SPEED_VARIATION_FACTOR_MAX = 1.02


class SimulationTUI(App):
    TITLE = "Federated SatComSim TUI Launcher"
    CSS_PATH = "tui_layout.css"  # External CSS file

    BINDINGS = [
        Binding("q", "quit_app", "Quit TUI & Services", show=True, priority=True),
        Binding("ctrl+c", "quit_app", "Quit TUI & Services", show=False, priority=True),
    ]

    status_message = reactive("Initializing...")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.running_processes = (
            {}
        )  # {name: {"proc": asyncio.subprocess.Process, "log_widget": Log}}
        self.core_ready_event = asyncio.Event()
        self.satellite_configs_for_core = []  # To store generated satellite configs

    def _generate_satellite_config(self, index, base_port):  # Renamed for clarity
        sat_id = f"TUI_Sat_{index:02d}"
        port = base_port + index

        # 1. Randomize Orbital Elements
        alt_km = random.uniform(MIN_ALTITUDE_KM, MAX_ALTITUDE_KM)
        a_km = EARTH_RADIUS_KM + alt_km  # Semi-major axis for a circular orbit (km)
        e = 0.0  # Eccentricity (circular orbit for simplicity)

        i_deg = random.uniform(0, 180.0)  # Inclination (0-180 degrees)
        raan_deg = random.uniform(
            0, 360.0
        )  # Right Ascension of Ascending Node (degrees)
        aop_deg = random.uniform(0, 360.0)  # Argument of Perigee (degrees)
        ta_deg = random.uniform(
            0, 360.0
        )  # True Anomaly (initial position in orbit, degrees)

        i_rad = np.radians(i_deg)
        raan_rad = np.radians(raan_deg)
        aop_rad = np.radians(aop_deg)
        ta_rad = np.radians(ta_deg)

        # 2. Calculate State Vector in Perifocal (Orbital) Frame
        r_mag_km = a_km  # For circular orbit, radius is semi-major axis

        # Position in perifocal frame (P, Q axes; W is normal to orbit so r_w = 0)
        r_perifocal_x = r_mag_km * np.cos(ta_rad)
        r_perifocal_y = r_mag_km * np.sin(ta_rad)
        r_perifocal_z = 0.0
        # r_perifocal_np = np.array([r_perifocal_x, r_perifocal_y, r_perifocal_z]) # For reference

        # Velocity in perifocal frame for circular orbit
        v_orbital_mag_km_s = np.sqrt(MU_EARTH_KM3_S2 / r_mag_km)
        v_perifocal_x = -v_orbital_mag_km_s * np.sin(ta_rad)
        v_perifocal_y = v_orbital_mag_km_s * np.cos(ta_rad)
        v_perifocal_z = 0.0
        # v_perifocal_np = np.array([v_perifocal_x, v_perifocal_y, v_perifocal_z]) # For reference

        # 3. Transformation Matrix from Perifocal to ECI
        # Using Z-X-Z Euler sequence for RAAN (Omega), Inclination (i), AOP (omega_small)
        # R = Rz(-RAAN) * Rx(-i) * Rz(-AOP)
        # Components of this matrix (from Vallado, "Fundamentals of Astrodynamics and Applications", or similar):
        # Px, Py, Pz are components of the Perifocal X-axis (P-axis) in ECI frame
        # Qx, Qy, Qz are components of the Perifocal Y-axis (Q-axis) in ECI frame

        cos_raan = np.cos(raan_rad)
        sin_raan = np.sin(raan_rad)
        cos_i = np.cos(i_rad)
        sin_i = np.sin(i_rad)
        cos_aop = np.cos(aop_rad)
        sin_aop = np.sin(aop_rad)

        # Direction cosines for P-vector in ECI
        Px = cos_raan * cos_aop - sin_raan * sin_aop * cos_i
        Py = sin_raan * cos_aop + cos_raan * sin_aop * cos_i
        Pz = sin_aop * sin_i

        # Direction cosines for Q-vector in ECI
        Qx = -cos_raan * sin_aop - sin_raan * cos_aop * cos_i
        Qy = -sin_raan * sin_aop + cos_raan * cos_aop * cos_i
        Qz = cos_aop * sin_i

        # Position in ECI (r_eci = r_perifocal_x * P_vec + r_perifocal_y * Q_vec)
        r_eci_x = r_perifocal_x * Px + r_perifocal_y * Qx
        r_eci_y = r_perifocal_x * Py + r_perifocal_y * Qy
        r_eci_z = r_perifocal_x * Pz + r_perifocal_y * Qz

        # Velocity in ECI (v_eci = v_perifocal_x * P_vec + v_perifocal_y * Q_vec)
        v_eci_x = v_perifocal_x * Px + v_perifocal_y * Qx
        v_eci_y = v_perifocal_x * Py + v_perifocal_y * Qy
        v_eci_z = v_perifocal_x * Pz + v_perifocal_y * Qz

        return {
            "id": sat_id,
            "host_for_service": SATELLITE_HOST_FOR_SERVICE,
            "port_for_service": port,
            "host_for_core": SATELLITE_HOST_FOR_CORE,
            "initial_state_vector": [
                round(r_eci_x, 3),
                round(r_eci_y, 3),
                round(r_eci_z, 3),
                round(v_eci_x, 3),
                round(v_eci_y, 3),
                round(v_eci_z, 3),
            ],
        }

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main_area"):
            with Vertical(classes="column", id="col_primary_services"):
                yield Label("Simulation Core:")
                yield Log(id="core_log", highlight=True, auto_scroll=True)
                yield Label(f"Groundstation ({GS_ID_TUI}):")
                yield Log(id="gs_log", highlight=True, auto_scroll=True)
                yield Label("Visualizer Console:")
                yield Log(id="vis_log", highlight=True, auto_scroll=True)

            with ScrollableContainer(classes="column", id="sat_logs_column"):
                yield Label(f"Satellite Logs ({NUM_SATELLITES_TUI} total):")
                # Satellite logs will be added dynamically here

            with Vertical(
                classes="column", id="col_right_stack"
            ):  # New parent for right column
                with Vertical(
                    id="gs_interaction_pane", classes="right_pane_section"
                ):  # Top half
                    yield Label(f"Interact with {GS_ID_TUI}:")
                    yield Log(id="gs_interaction_log", highlight=True, auto_scroll=True)
                    yield Input(
                        placeholder=f"Cmd for {GS_ID_TUI} (e.g., LIST_VISIBLE)",
                        id="gs_command_input",
                    )
                    yield Button("Send to GS", id="gs_send_button", variant="primary")
                with Vertical(
                    id="placeholder_pane", classes="right_pane_section"
                ):  # Bottom half
                    yield Label("Information / Status Pane:")
                    yield Static(
                        "The VPython visualizer runs in its own separate window.\n\n"
                        "This TUI is for the simulation control console.\n\n",
                        id="info_placeholder_text",
                    )
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one(Footer).status_text = "Starting simulation services..."
        asyncio.create_task(self.manage_simulation_lifecycle())

    async def manage_simulation_lifecycle(self):
        core_log = self.query_one("#core_log", Log)
        gs_log = self.query_one("#gs_log", Log)
        vis_log = self.query_one("#vis_log", Log)
        sat_logs_container = self.query_one("#sat_logs_column", ScrollableContainer)

        core_log.write_line(
            "TEST: manage_simulation_lifecycle started, core_log obtained."
        )

        try:
            # 1. Launch Core
            core_cmd = [PYTHON_EXECUTABLE, "-u", CORE_SCRIPT]
            await self.launch_service("Core", core_cmd, core_log)
            core_log.write_line(
                "Waiting for Core to initialize (e.g., control port)..."
            )
            await asyncio.sleep(7)  # Adjust as needed; robust check would be better
            self.core_ready_event.set()
            core_log.write_line("Core assumed ready.")

            vis_cmd = [PYTHON_EXECUTABLE, "-u", VISUALIZER_SCRIPT]
            await self.launch_service("Visualizer", vis_cmd, vis_log)

            # 2. Generate and Launch Satellite Services
            self.satellite_configs_for_core = []
            for i in range(NUM_SATELLITES_TUI):
                sat_config = self._generate_satellite_config(
                    i, SATELLITE_SERVICE_BASE_PORT
                )
                self.satellite_configs_for_core.append(sat_config)

                sat_log_label = Label(
                    f"Log: {sat_config['id']} (Port: {sat_config['port_for_service']})"
                )
                sat_log_widget = Log(
                    highlight=True, auto_scroll=True, classes="satellite_log_instance"
                )

                await sat_logs_container.mount(sat_log_label)
                await sat_logs_container.mount(sat_log_widget)

                sat_cmd = [
                    PYTHON_EXECUTABLE,
                    "-u",
                    SATELLITE_SCRIPT,
                    "--id",
                    sat_config["id"],
                    "--host",
                    sat_config["host_for_service"],
                    "--port",
                    str(sat_config["port_for_service"]),
                ]
                await self.launch_service(sat_config["id"], sat_cmd, sat_log_widget)
            sat_logs_container.scroll_y = 0  # Scroll sat logs to top

            core_log.write_line("Waiting a bit for all satellite services to start...")
            await asyncio.sleep(NUM_SATELLITES_TUI * 0.4 + 1)  # Scaled delay

            # 3. Add Satellites to Core
            await self.core_ready_event.wait()  # Should already be set
            if self.satellite_configs_for_core:
                core_log.write_line(
                    f"Attempting to add {len(self.satellite_configs_for_core)} satellites to core..."
                )
            for sat_config in self.satellite_configs_for_core:
                add_payload = {
                    "id": sat_config["id"],
                    "host": sat_config["host_for_core"],
                    "port": sat_config["port_for_service"],
                    "initial_state_vector": sat_config["initial_state_vector"],
                }
                add_cmd_str = f"ADD_SATELLITE {json.dumps(add_payload)}\n"
                response = await self.send_command_to_tcp_server(
                    "CoreControl",
                    CORE_CONTROL_HOST,
                    CORE_CONTROL_PORT,
                    add_cmd_str,
                    core_log,
                )
                core_log.write_line(f"ADD_SATELLITE for {sat_config['id']}: {response}")
                await asyncio.sleep(0.1)  # Stagger ADD commands slightly

            # 4. Launch Groundstation Service
            gs_cmd = [
                PYTHON_EXECUTABLE,
                "-u",
                GROUNDSTATION_SCRIPT,
                "--id",
                GS_ID_TUI,
                "--core-host",
                GS_CORE_HOST_TUI,
                "--core-port",
                str(GS_CORE_PORT_TUI),
                "--listen-host",
                "localhost",
                "--listen-port",
                str(GS_USER_CMD_PORT_TUI),
            ]
            await self.launch_service(GS_ID_TUI, gs_cmd, gs_log)
            await asyncio.sleep(1.5)

            self.query_one(Footer).status_text = (
                "All services initiated. Press Q or Ctrl+C to Quit."
            )
        except Exception as e:
            self.query_one(Footer).status_text = (
                f"Error during lifecycle management: {e}"
            )
            core_log.write_line(f"FATAL ERROR in manage_simulation_lifecycle: {e}")

    async def launch_service(self, name: str, command: list, log_widget: Log):
        log_widget.write_line(f"--- Launching {name}: {' '.join(command)} ---")
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Combine stderr with stdout
            )
            self.running_processes[name] = {"proc": process, "log_widget": log_widget}
            log_widget.write_line(f"{name} started (PID: {process.pid})...")
            asyncio.create_task(
                self._read_stream(process.stdout, log_widget, name, "LOG")
            )
            asyncio.create_task(self._monitor_process(process, name, log_widget))
        except Exception as e:
            log_widget.write_line(f"--- Failed to start {name}: {e} ---")
            self.running_processes.pop(name, None)  # Remove if launch failed

    async def _read_stream(
        self,
        stream: asyncio.StreamReader,
        log_widget: Log,
        service_name: str,
        stream_name: str,
    ):
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                # Call a method to update widget from main thread if Textual requires it
                # For Log widget, write_line is thread-safe if called via call_from_thread or from an async method scheduled by Textual
                log_widget.write_line(f"{line.decode(errors='ignore').strip()}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log_widget.write_line(
                f"--- Error reading {service_name} ({stream_name}): {e} ---"
            )
        finally:
            log_widget.write_line(
                f"--- {service_name} ({stream_name}) stream ended ---"
            )

    async def _monitor_process(
        self, process: asyncio.subprocess.Process, name: str, log_widget: Log
    ):
        await process.wait()
        log_widget.write_line(
            f"--- Service {name} (PID: {process.pid}) terminated with exit code {process.returncode}. ---"
        )
        self.running_processes.pop(name, None)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "gs_send_button":
            input_widget = self.query_one("#gs_command_input", Input)
            command_text = input_widget.value
            gs_interaction_log = self.query_one("#gs_interaction_log", Log)
            if command_text:
                gs_interaction_log.write_line(f"> {command_text}")
                response = await self.send_command_to_tcp_server(
                    GS_ID_TUI,
                    "localhost",
                    GS_USER_CMD_PORT_TUI,
                    command_text + "\n",
                    gs_interaction_log,
                )
                gs_interaction_log.write_line(f"< {response}")
                input_widget.value = ""

    async def send_command_to_tcp_server(
        self,
        service_name_for_log: str,
        host: str,
        port: int,
        command: str,
        log_widget_ref: Log,
    ):
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=5.0
            )
            writer.write(command.encode())
            await writer.drain()
            response_data = await asyncio.wait_for(
                reader.readuntil(b"\n"), timeout=10.0
            )
            response = response_data.decode(errors="ignore").strip()
            writer.close()
            await writer.wait_closed()
            return response
        except asyncio.TimeoutError:
            msg = f"Timeout with {service_name_for_log} ({host}:{port})."
        except ConnectionRefusedError:
            msg = f"Connection refused by {service_name_for_log} ({host}:{port})."
        except Exception as e:
            msg = f"Error with {service_name_for_log} ({host}:{port}): {e}"
        log_widget_ref.write_line(f"TCP CMD ERROR: {msg}")  # Log error to relevant log
        return msg

    async def action_quit_app(self) -> None:
        self.query_one(Footer).status_text = "Shutting down all services..."
        self.bell()

        # Create a list of processes to terminate
        procs_to_terminate = []
        for name, info in list(self.running_processes.items()):
            if info["proc"].returncode is None:  # Process still running
                procs_to_terminate.append((name, info["proc"], info["log_widget"]))

        if not procs_to_terminate:
            self.exit("No active processes to terminate. Exiting.")
            return

        for name, proc, log_widget in procs_to_terminate:
            log_widget.write_line(
                f"--- Sending SIGTERM to {name} (PID: {proc.pid}) ---"
            )
            try:
                proc.terminate()
            except ProcessLookupError:
                log_widget.write_line(f"--- {name} already gone (terminate) ---")

        # Give processes a moment to terminate gracefully
        shutdown_message_area = self.query_one(Footer)  # Or a dedicated status widget
        for i in range(5, 0, -1):
            shutdown_message_area.status_text = (
                f"Waiting for services to close ({i}s)..."
            )
            await asyncio.sleep(1)

        for name, proc, log_widget in procs_to_terminate:
            if proc.returncode is None:  # Still running after SIGTERM and wait
                log_widget.write_line(
                    f"--- Sending SIGKILL to {name} (PID: {proc.pid}) ---"
                )
                try:
                    proc.kill()
                except ProcessLookupError:
                    log_widget.write_line(f"--- {name} already gone (kill) ---")
            else:
                log_widget.write_line(
                    f"--- {name} terminated with code {proc.returncode} ---"
                )

        self.running_processes.clear()
        self.exit("Simulation services shut down.")


if __name__ == "__main__":
    # Create log directory if it doesn't exist, before Textual app starts
    os.makedirs(LOG_DIR_TUI, exist_ok=True)
    app = SimulationTUI()
    app.run()
