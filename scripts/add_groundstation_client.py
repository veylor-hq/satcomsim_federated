import socket
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Client to send ADD_GROUNDSTATION command to Simulation Core"
    )

    # Groundstation specific arguments
    parser.add_argument(
        "--id",
        type=str,
        required=True,
        help="Unique Groundstation ID (e.g., GS_London)",
    )
    parser.add_argument(
        "--lat",
        type=float,
        required=True,
        help="Latitude of the Groundstation (degrees)",
    )
    parser.add_argument(
        "--lon",
        type=float,
        required=True,
        help="Longitude of the Groundstation (degrees)",
    )
    parser.add_argument(
        "--alt",
        type=float,
        default=0.0,
        help="Altitude of the Groundstation (km, default: 0.0)",
    )
    parser.add_argument(
        "--min-elev",
        type=float,
        default=5.0,
        help="Minimum elevation mask for visibility (degrees, default: 5.0)",
    )

    # Core connection arguments
    parser.add_argument(
        "--core-host",
        type=str,
        default="localhost",
        help="Host of the Simulation Core Control Service (default: localhost)",
    )
    parser.add_argument(
        "--core-port",
        type=int,
        default=60000,
        help="Port of the Simulation Core Control Service (default: 60000)",
    )

    args = parser.parse_args()

    # Construct the groundstation configuration dictionary
    new_groundstation_config = {
        "id": args.id,
        "lat_deg": args.lat,
        "lon_deg": args.lon,
        "alt_km": args.alt,
        "min_elevation_deg": args.min_elev,
    }

    # Construct the command string
    command_payload_str = json.dumps(new_groundstation_config)
    command = f"ADD_GROUNDSTATION {command_payload_str}\n"  # Ensure newline for server parsing

    print(f"Attempting to send command to core at {args.core_host}:{args.core_port}")
    print(f"Command: {command.strip()}")  # Print command without newline for display

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(10.0)  # Set a timeout for connection and operations
            s.connect((args.core_host, args.core_port))
            print(f"Connected to Simulation Core.")

            s.sendall(command.encode("utf-8"))
            print("Command sent.")

            response_bytes = s.recv(1024)  # Wait for a response
            response_str = response_bytes.decode("utf-8").strip()
            print(f"Response from core: {response_str}")

    except socket.timeout:
        print(
            f"Error: Timeout connecting to or communicating with {args.core_host}:{args.core_port}"
        )
    except ConnectionRefusedError:
        print(
            f"Error: Connection refused by {args.core_host}:{args.core_port}. Is the core running and listening?"
        )
    except Exception as e:
        print(f"An error occurred: {e}")
