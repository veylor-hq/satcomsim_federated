import asyncio
import json
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class Satellite:
    def __init__(self, satellite_id="SatDefault", host="localhost", port=65432):
        self.satellite_id = satellite_id
        self.host = host
        self.port = port
        self.current_state_from_core = (
            {}
        )  # Store state received from the simulation core
        self.logger = logging.getLogger(f"SatelliteService[{self.satellite_id}]")
        self.logger.info(f"Initializing to listen on {self.host}:{self.port}...")

    def _parse_json_message(self, data_bytes: bytes) -> tuple[dict | None, bytes]:
        """
        Tries to parse a JSON message from the beginning of data_bytes.
        Assumes messages are newline-terminated JSON strings.
        Returns (parsed_json_dict, remaining_bytes_buffer) or (None, original_data_bytes)
        """
        try:
            # Find the first newline, which should terminate a complete JSON message from core/gs
            newline_index = data_bytes.find(b"\n")
            if newline_index != -1:
                json_str = data_bytes[:newline_index].decode("utf-8", errors="ignore")
                remaining_buffer = data_bytes[newline_index + 1 :]
                message = json.loads(json_str)
                return message, remaining_buffer
        except json.JSONDecodeError as jde:
            self.logger.error(
                f"JSON Decode Error: {jde} for data: '{data_bytes[:200].decode(errors='ignore')}...'"
            )
            if newline_index != -1:
                return None, data_bytes[newline_index + 1 :]  # Discard corrupt segment
            return None, data_bytes  # Incomplete segment, keep buffering
        except Exception as e:
            self.logger.error(f"Error in _parse_json_message: {e}")
            if newline_index != -1:
                return None, data_bytes[newline_index + 1 :]
            return None, data_bytes
        return None, data_bytes  # No newline found, need more data

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        peer_address_tuple = writer.get_extra_info("peername")
        peer_address_str = (
            f"{peer_address_tuple[0]}:{peer_address_tuple[1]}"
            if peer_address_tuple
            else "Unknown Peer"
        )
        self.logger.info(f"Accepted connection from {peer_address_str}")

        buffer = b""
        try:
            while True:
                data_chunk = await reader.read(4096)  # Read up to 4KB
                if not data_chunk:
                    self.logger.info(f"Connection closed by {peer_address_str} (EOF).")
                    if buffer:
                        self.logger.warning(
                            f"Partial message in buffer at close: {buffer.decode(errors='ignore')}"
                        )
                    break

                buffer += data_chunk

                while True:  # Process all complete messages in the buffer
                    message_dict, remaining_buffer = self._parse_json_message(buffer)
                    buffer = remaining_buffer  # Update buffer with what's left

                    if message_dict:
                        self.logger.debug(f"Received message object: {message_dict}")
                        message_type = message_dict.get("type")
                        payload = message_dict.get("payload")

                        response_str = None  # For commands from groundstations

                        if message_type == "STATE_UPDATE":  # From Simulation Core
                            self.current_state_from_core = payload if payload else {}
                            ts = self.current_state_from_core.get(
                                "timestamp_sim_sec", "N/A"
                            )
                            pos = self.current_state_from_core.get(
                                "position_eci_km", "N/A"
                            )
                            vel = self.current_state_from_core.get(
                                "velocity_eci_km_s", "N/A"
                            )
                            self.logger.info(
                                f"Core State Update @ SimTime {ts}: Pos: {pos}, Vel: {vel}"
                            )
                            # This type of message (from core) usually doesn't require a response back to the core

                        elif (
                            message_type == "GROUND_COMMAND"
                        ):  # Example: From Groundstation
                            command_content = (
                                payload.get("command", "NO_COMMAND")
                                if payload
                                else "NO_COMMAND_PAYLOAD"
                            )
                            self.logger.info(
                                f"Received GROUND_COMMAND: '{command_content}'"
                            )
                            # Process the command here
                            # For now, just acknowledge it
                            response_str = f"ACK_GROUND_COMMAND: '{command_content}' received by {self.satellite_id}"

                        # Add more message_type handlers as needed (e.g., from other satellites)

                        else:
                            self.logger.warning(
                                f"Received unknown message type: '{message_type}' from {peer_address_str}"
                            )
                            response_str = f"NACK_UNKNOWN_TYPE: '{message_type}'"

                        # If a response is generated (typically for groundstation commands)
                        if response_str and writer and not writer.is_closing():
                            try:
                                self.logger.info(
                                    f"Sending response to {peer_address_str}: {response_str}"
                                )
                                writer.write((response_str + "\n").encode("utf-8"))
                                await writer.drain()
                            except (ConnectionResetError, BrokenPipeError):
                                self.logger.warning(
                                    f"Could not send response to {peer_address_str}, connection lost."
                                )
                                break  # Break from inner while True, will lead to outer while True checking reader.read
                            except Exception as e_write:
                                self.logger.error(
                                    f"Error sending response to {peer_address_str}: {e_write}"
                                )
                                break

                    else:  # No complete message parsed from buffer this iteration
                        break  # Break from inner while True, wait for more data from reader.read()

        except asyncio.CancelledError:
            self.logger.info(f"Client handler for {peer_address_str} cancelled.")
        except ConnectionResetError:
            self.logger.info(f"Connection reset by {peer_address_str}.")
        except asyncio.IncompleteReadError:  # Can happen if client disconnects abruptly
            self.logger.info(
                f"Incomplete read from {peer_address_str}, likely disconnected."
            )
        except Exception as e:
            self.logger.error(
                f"Connection error with {peer_address_str}: {e}", exc_info=True
            )  # Log traceback for unexpected errors
        finally:
            self.logger.info(f"Closing connection with {peer_address_str}")
            if writer and not writer.is_closing():
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception as e_close:
                    self.logger.error(
                        f"Error during writer close for {peer_address_str}: {e_close}"
                    )

    async def start_server(self):
        """Starts the TCP server for this satellite service."""
        try:
            server = await asyncio.start_server(
                self.handle_client, self.host, self.port
            )
            addr = (
                server.sockets[0].getsockname()
                if server.sockets
                else (self.host, self.port)
            )
            self.logger.info(f"Serving on {addr}")

            async with server:  # Ensures server.close() is called
                await server.serve_forever()
        except OSError as e:  # e.g., port already in use
            self.logger.error(
                f"Could not start server on {self.host}:{self.port}. Error: {e}"
            )
            # Re-raise or handle as appropriate for your application lifecycle
            raise
        except asyncio.CancelledError:
            self.logger.info("Server task cancelled.")
        finally:
            self.logger.info("Server shutting down.")
            # Server.close() and wait_closed() are handled by 'async with server' context manager


async def main(args):
    """Main coroutine to run the satellite service."""
    satellite_instance = Satellite(satellite_id=args.id, host=args.host, port=args.port)
    try:
        await satellite_instance.start_server()
    except asyncio.CancelledError:
        satellite_instance.logger.info("Satellite service main task cancelled.")
