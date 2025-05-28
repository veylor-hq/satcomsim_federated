import socket
import json

CORE_CONTROL_HOST = "localhost"
CORE_CONTROL_PORT = 60000  # Must match the core's control port

new_satellite_config = {
    "id": "SatGamma",
    "host": "localhost",
    "port": 65435,  # Port SatGamma service is listening on
    "initial_state_vector": [40000.0, 0.0, 0.0, 0.0, 3.5, 0.0],  # Example GEO-like
}

command = f"ADD_SATELLITE {json.dumps(new_satellite_config)}\n"

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((CORE_CONTROL_HOST, CORE_CONTROL_PORT))
        s.sendall(command.encode())
        response = s.recv(1024)
        print(f"Response from core: {response.decode()}")
except Exception as e:
    print(f"Error sending command: {e}")
