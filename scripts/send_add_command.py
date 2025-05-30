import socket
import json
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Federated Satellite Service")
    parser.add_argument(
        "--id", type=str, default="SatDefault", help="Unique Satellite ID"
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host of the Satellite Service"
    )

    parser.add_argument(
        "--port", type=int, default=65432, help="Port of the Satellite Service"
    )

    parser.add_argument(
        "--initial-state-vector",
        nargs="+",
        type=float,
        default=[4414.119, -9.847, 5518.923, -5.865, 0.013, 4.691],
        help="Initial State Vector for the Satellite",
    )

    parser.add_argument(
        "--core-host",
        type=str,
        default="localhost",
        help="Host of the Core Control Service",
    )
    parser.add_argument(
        "--core-port", type=int, default=60000, help="Port of the Core Control Service"
    )

    args = parser.parse_args()

    new_satellite_config = {
        "id": args.id,
        "host": args.host,
        "port": args.port,
        "initial_state_vector": args.initial_state_vector,
    }

    command = f"ADD_SATELLITE {json.dumps(new_satellite_config)}\n"

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((args.core_host, args.core_port))
            s.sendall(command.encode())
            response = s.recv(1024)
            print(f"Response from core: {response.decode()}")
    except Exception as e:
        print(f"Error sending command: {e}")
