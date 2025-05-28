# gs_interaction_client.py
import socket
import sys
import argparse

DEFAULT_GS_HOST = "localhost"


def main(gs_host, gs_port):
    print(f"Attempting to connect to Groundstation at {gs_host}:{gs_port}...")
    client_socket = None  # Initialize to ensure it's defined for finally block

    try:
        # Create a TCP/IP socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(15.0)  # Timeout for connection and receive operations

        # Connect the socket to the server's address and port
        client_socket.connect((gs_host, gs_port))
        print(f"Successfully connected to Groundstation {gs_host}:{gs_port}")
        print("\nAvailable commands:")
        print(
            "  LIST_VISIBLE                     - Lists satellites currently visible to this groundstation."
        )
        print(
            "  SEND <satellite_id> <message>    - Sends a message to a visible satellite."
        )
        print(
            "                                     (e.g., SEND SatAlpha_LEO GET_STATUS)"
        )
        print(
            "  QUIT                             - Disconnects from this groundstation service (server closes this specific client connection)."
        )
        print("  exit                             - Closes this client application.")
        print("-" * 40)

        while True:
            try:
                user_input = input(f"GS ({gs_host}:{gs_port})> ")
                if not user_input.strip():  # Handle empty input
                    continue

                if user_input.strip().lower() == "exit":
                    print("Exiting client application...")
                    break

                # Send data (add newline as the server expects it for readuntil)
                command_to_send = user_input + "\n"
                client_socket.sendall(command_to_send.encode("utf-8"))

                # If user types QUIT for the server, the server might just close or send an ACK
                if user_input.strip().upper() == "QUIT":
                    print(
                        "Sent QUIT to groundstation service. Waiting for confirmation/disconnection..."
                    )
                    try:
                        # Try to receive a final message or detect closure
                        response = client_socket.recv(4096)
                        if response:
                            print(f"GS Response: {response.decode('utf-8').strip()}")
                        else:
                            print(
                                "Groundstation closed connection as expected after QUIT."
                            )
                    except socket.timeout:
                        print(
                            "Timeout waiting for response to QUIT. Assuming disconnection."
                        )
                    except (ConnectionResetError, BrokenPipeError):
                        print("Connection reset by server after QUIT.")
                    except Exception as e_quit_recv:
                        print(
                            f"Note: Error during receive after QUIT (server might have closed immediately): {e_quit_recv}"
                        )
                    break  # Exit client loop after sending QUIT

                # Look for the response from other commands
                response_data = client_socket.recv(4096)  # Buffer size for response
                if response_data:
                    print(f"GS Response: {response_data.decode('utf-8').strip()}")
                else:
                    # This means the server closed the connection unexpectedly
                    print("Connection unexpectedly closed by groundstation service.")
                    break

            except KeyboardInterrupt:
                print(
                    "\nCtrl+C detected. Sending QUIT to groundstation and exiting client."
                )
                if client_socket:
                    try:
                        client_socket.sendall(b"QUIT\n")
                        # Short wait for server to process QUIT before client closes
                        time.sleep(0.1)
                    except Exception as e_send_quit:
                        print(f"Error sending QUIT on Ctrl+C: {e_send_quit}")
                break  # Exit main loop
            except socket.timeout:
                print(
                    "Socket operation timed out. Connection might be lost. Please try again or 'exit'."
                )
                # Optionally, break here or allow retrying
                # break
            except (socket.error, BrokenPipeError, ConnectionResetError) as e:
                print(f"Socket error: {e}. Connection lost.")
                break
            except Exception as e:
                print(f"An unexpected client-side error occurred: {e}")
                break  # Exit on other unexpected errors

    except socket.timeout:
        print(f"Connection attempt to {gs_host}:{gs_port} timed out.")
    except ConnectionRefusedError:
        print(
            f"Connection to {gs_host}:{gs_port} refused. Please ensure the groundstation service is running and listening on that port."
        )
    except Exception as e_conn:
        print(f"Failed to connect to {gs_host}:{gs_port}. Error: {e_conn}")
    finally:
        if client_socket:
            print("Closing client socket.")
            client_socket.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Client to interact with a SatComSim Groundstation Service."
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_GS_HOST,
        help=f"Groundstation service host (default: {DEFAULT_GS_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Groundstation service port (e.g., 7000 for GS_London, 7001 for GS_NewYork)",
    )

    args = parser.parse_args()

    # Import time here if not already imported, for the Ctrl+C sleep
    import time

    main(args.host, args.port)
