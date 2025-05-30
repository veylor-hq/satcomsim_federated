# Federated Satellite Communication Simulator (SatComSim_Federated)
<a href="https://www.youtube.com/watch?v=M2PyWJDtF58" target="_blank"><img src="http://img.youtube.com/vi/M2PyWJDtF58/0.jpg"
alt="M2PyWJDtF58" width="240" height="180" border="10" /></a>

## 🛰️ Overview

SatComSim_Federated is a Python-based simulation environment designed to model and simulate a constellation of satellites, groundstations, and their interactions in Earth orbit. It features a federated architecture where different components of the simulation (core physics engine, individual satellite behaviors, groundstation logic, visualization) run as separate, cooperating services communicating over TCP/IP.

This project aims to provide a flexible platform for testing satellite communication protocols, visibility windows, groundstation operations, dynamic constellation management, and basic orbital mechanics visualization.

### Small Note  
Theoretically, Groundstation and Satellite services could be(or will be) moved to a separate repository, allowing for a more modular architecture. However, for simplicity and ease of development at current stage, they are currently included in the same repository as the core simulation logic.

## ✨ Features

* **Federated Architecture:** Core simulation logic, satellites, and groundstations run as independent services.
* **Dynamic Entity Management:** Satellites and groundstations can be added to the simulation dynamically during runtime via control commands.
* **Orbital Mechanics:**
    * Basic 2-body problem orbit propagation using Runge-Kutta 4th order (RK4) integrator.
    * Support for Earth-Centered Inertial (ECI) coordinate frame.
* **Coordinate Transformations:** Includes LLA (Latitude, Longitude, Altitude), ECEF (Earth-Centered, Earth-Fixed), and ECI transformations.
* **Earth Rotation:** Simulates Earth's rotation for accurate groundstation positioning and visibility calculations.
* **Groundstation Visibility:**
    * Calculates satellite visibility from groundstations based on ECI positions and a configurable minimum elevation mask.
    * Uses geodetic "up" vector for more accurate local horizon definition.
* **Decoupled Onboard Software:** Each satellite service (Bus OS) launches a separate "onboard software" process, allowing for modular and potentially language-independent mission logic. Communication is via local TCP.
* **Communication Protocol:** Services communicate using newline-delimited JSON messages over TCP/IP.
* **3D Visualization (`visualizer_vpython.py`):**
    * Real-time display of Earth, ECI axes, satellite positions, groundstation positions.
    * Satellite trails and predicted orbital paths.
    * Interactive features: click on satellites for details, toggle orbit predictions.
    * Rotating Earth texture.
* **Text User Interface (`tui_launcher.py`):**
    * A [Textual](https://github.com/Textualize/textual) TUI to launch, manage, and monitor all simulation services.
    * Tiled layout displaying real-time logs from each component.
    * Integrated panel for sending commands to groundstation services.
    * Graceful shutdown of all managed processes.
* **Client Scripts:** Utility scripts for adding satellites/groundstations and interacting with groundstation services.

## 🏛️ Architecture

The simulation suite is composed of several key Python services:

1.  **Simulation Core (`simulation_core/` package):**
    * The central brain. Manages simulation time, global truth for ECI frames, and Earth's state.
    * Propagates satellite orbits.
    * Calculates groundstation ECI positions and satellite visibility.
    * Listens for control commands (e.g., `ADD_SATELLITE`, `ADD_GROUNDSTATION`).
    * Streams data to visualizers and groundstation services.
2.  **Satellite Service (`satellite_service.py`):**
    * Represents the "Bus OS" for an individual satellite.
    * Receives its ECI state from the Core.
    * Launches and manages its own `onboard_software_service.py` process.
    * Acts as a TCP server for commands from the Core (state updates) and Groundstations (relayed to onboard software).
3.  **Onboard Software Service (`onboard_software_service.py`):**
    * A separate process launched by a `satellite_service.py` instance.
    * Contains the specific mission logic, command processing, and "intelligence" for that satellite.
    * Communicates with its parent `satellite_service.py` via a local TCP connection.
4.  **Groundstation Service (`groundstation_service.py`):**
    * Represents a fixed groundstation on Earth.
    * Connects to the Simulation Core to get its own ECI position and a list of currently visible satellites (including their direct connection details).
    * Runs a TCP server to accept commands from a user/client (e.g., `gs_interaction_client.py`).
    * Forwards commands to visible satellites by connecting directly to the respective `satellite_service.py`.
5.  **Visualizer (`visualizer_vpython.py`):**
    * A VPython application that connects to the Simulation Core for state data and renders the 3D scene. Runs in its own graphical window.
6.  **TUI Launcher (`tui_launcher.py`):**
    * A Textual application that orchestrates the launching of all other services and displays their console logs in a unified interface.

**Communication Flow (Simplified):**
* Core calculates and sends ECI state to Satellite Services.
* Core calculates and sends ECI state + visibility data to Groundstation Services.
* Core calculates and sends ECI states of all objects + visibility links to Visualizer.
* Users interact with Groundstation Services (via TUI panel or `gs_interaction_client.py`).
* Groundstation Services send commands to visible Satellite Services.
* Satellite Services (Bus OS) forward these commands to their Onboard Software process.
* Onboard Software processes commands and sends responses back up the chain.

---
## 🚀 Getting Started

### Prerequisites

* Python 3.10+ (asyncio features are used extensively).
* `pip` for installing packages or `poetry`.
* (Optional but Recommended) A virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate   # Windows
    ```

### Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/veylor-hq/satcomsim_federated.git
    cd satcomsim_federated
    ```

2.  **Install required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```  
    or if using Poetry:
    ```bash  
    poetry install
    ```

---
## 🚦 How to Run

The primary way to run the full simulation suite is using the Textual TUI Launcher.

**1. Using the TUI Launcher (`tui_launcher.py`)**

   This script will start the Simulation Core, Visualizer (console output), multiple Satellite Services, the Onboard Software for each satellite, and Groundstation Services, displaying their logs in a tiled interface. It also provides a panel for interacting with a groundstation.

   * **Run from the project root directory (`satcomsim_federated`):**
       ```bash
       python tui/tui_launcher.py
       ```
   * The TUI will appear in your terminal. Services will be launched automatically.
   * The VPython visualizer will open in its own separate graphical window.
   * Use the "Groundstation Interaction" panel in the TUI to send commands (e.g., `LIST_VISIBLE` to GS_London, or `SEND <SatID> {"task": "GET_STATUS"}`).
   * Press `q` or `Ctrl+C` in the TUI window to gracefully shut down all launched services.

**2. Running Individual Test/Utility Scripts**

   For specific testing or direct interaction, you can run other scripts (ensure the Simulation Core is running first if they depend on it):
   * **Add a satellite dynamically:**
       ```bash
       # Make sure simulation_core.py is running.
       # Make sure a satellite_service.py instance is (or will be) running on the specified port.
       python scripts/add_satellite_client.py --id MySat1 --host localhost --port 65432 --initial-state-vector 7000 0 0 0 7.6 0
       ```
   * **Add a groundstation dynamically:**
       ```bash
       # Make sure simulation_core.py is running.
       python scripts/add_groundstation_client.py --id GS_Paris --lat 48.85 --lon 2.35 --alt 0.04
       ```
   * **Interact with a Groundstation Service:**
       ```bash
       # Make sure simulation_core.py is running.
       # Make sure the target groundstation_service.py (e.g., for GS_London on port 7000) is running.
       python gs_interaction_client.py --port 7000 
       ```

---
## ⚙️ How It Works (Technical Details)

* **Simulation Core (`simulation_core` package):**
    * The `SimulationCore` class in `core_logic.py` is the heart.
    * It uses an `asyncio` event loop to manage time steps and concurrent operations.
    * `run_simulation_loop()`: Advances simulation time, updates all ECI positions (satellites via RK4 propagation using imported `eom_func` and `rk4_step_func`; groundstations via Earth rotation using imported transformation functions).
    * It hosts multiple `asyncio.start_server` instances for:
        * Control commands (dynamic addition of entities).
        * Streaming data to visualizers (`VIS_UPDATE` messages).
        * Streaming data to groundstation services (`GS_SIM_UPDATE` messages, including visibility lists and GS ECI positions).
    * Visibility is calculated in `calculate_visibility()` using dot products and elevation angles, considering the groundstation's geodetic "up" vector (transformed to ECI).
    * Communication with satellite services is client-to-server (core sends `STATE_UPDATE`).
* **Satellite Service (`satellite_service.py`):**
    * Acts as a "Bus OS." Launches an `onboard_software_service.py` as a subprocess.
    * Runs an `asyncio` TCP server.
    * Receives `STATE_UPDATE` from the core.
    * Receives `GROUND_COMMAND` from a groundstation, forwards its payload to its `onboard_software_service` via a local TCP client connection.
    * Relays the response from onboard software back to the groundstation.
* **Onboard Software Service (`onboard_software_service.py`):**
    * A standalone `asyncio` TCP server launched by `satellite_service.py`.
    * Listens for commands (JSON) from its parent bus.
    * Contains placeholder logic for tasks like `GET_STATUS`, `SET_PAYLOAD_MODE`. This is where custom satellite behavior is implemented.
    * Sends JSON responses back to the bus.
* **Groundstation Service (`groundstation_service.py`):**
    * Connects as a client to the Simulation Core's GS update stream.
    * Parses `GS_SIM_UPDATE` to find its own data and list of visible satellites (including satellite `host:port` for direct comms).
    * Runs an `asyncio` TCP server for user commands (e.g., from `gs_interaction_client.py` or the TUI).
    * If commanded to `SEND <sat_id> <payload>`:
        * Checks if `<sat_id>` is in its `visible_satellites_info`.
        * If yes, connects as a TCP client to that satellite's `satellite_service.py` and sends the payload.
        * Receives and relays the satellite's response.
* **Message Passing:** All inter-service TCP communication uses newline-terminated JSON strings for messages, typically structured with a `"type"` and `"payload"` field.

---
## 🛠️ Structure of the Codebase 

## 🔮 Future Enhancements / To-Do  

* Implement more sophisticated orbital force models (J2, atmospheric drag, SRP).
* Develop detailed payload models and commands for the `onboard_software_service.py`.
* Implement bi-directional telemetry streams from satellites to groundstations.
* Add resource management (power, fuel) to satellites.
* Enhance groundstation models (antenna patterns, pointing).
* Improve error handling and automatic reconnection for all TCP links.
* Load initial satellite/groundstation configurations from external files (e.g., JSON, YAML) instead of hardcoding in launcher scripts.
* More advanced status reporting and control in the TUI.
* Consider Dockerization for easier deployment and dependency management of services.
