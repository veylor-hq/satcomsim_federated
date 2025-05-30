# onboard_software_service.py
import asyncio
import json
import argparse
import logging
import random  # For simulating some actions
import time  # For simulating work

# Setup basic logging for this service
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s][%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ],  # Ensure logs go to stdout/stderr for capture by parent
)

MAX_BUFFER_SIZE = 1024 * 1024  # 1MB, to prevent unbounded buffer growth


class OnboardSoftware:
    def __init__(self, software_id, host, port):
        self.software_id = software_id
        self.host = host
        self.port = port
        self.logger = logging.getLogger(f"OnboardSW[{self.software_id}]")

        self.power_status = "NOMINAL"
        self.payload_mode = "IDLE"
        self.current_attitude = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self.telemetry_counter = 0

    async def handle_bus_command(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        peer_address_tuple = writer.get_extra_info("peername")
        peer_address_str = (
            f"{peer_address_tuple[0]}:{peer_address_tuple[1]}"
            if peer_address_tuple
            else "Unknown Peer"
        )
        self.logger.info(f"Accepted connection from Satellite Bus: {peer_address_str}")

        buffer = b""
        try:
            while True:  # Outer loop for reading from socket
                data_chunk = await reader.read(4096)
                if not data_chunk:
                    self.logger.info(
                        f"Satellite Bus {peer_address_str} closed connection (EOF)."
                    )
                    if buffer:
                        self.logger.warning(
                            f"Partial message in buffer at close: {buffer.decode(errors='ignore')}"
                        )
                    break  # Break outer loop

                buffer += data_chunk

                if len(buffer) > MAX_BUFFER_SIZE:
                    self.logger.error(
                        f"Buffer size exceeded {MAX_BUFFER_SIZE}. Clearing buffer and closing connection."
                    )
                    # Optionally send an error message before closing if possible
                    try:
                        err_resp = (
                            json.dumps(
                                {
                                    "status": "ERROR",
                                    "message": "Buffer overflow on OSW side.",
                                }
                            )
                            + "\n"
                        )
                        writer.write(err_resp.encode("utf-8"))
                        await writer.drain()
                    except:  # pylint: disable=bare-except
                        pass
                    break  # Break outer loop, connection will be closed in finally

                # Inner loop to process all complete newline-terminated messages in the buffer
                while True:
                    newline_index = buffer.find(b"\n")
                    if newline_index == -1:  # No complete message in buffer yet
                        break  # Break inner loop, go back to read more data_chunk from socket

                    message_bytes = buffer[:newline_index]
                    buffer = buffer[
                        newline_index + 1 :
                    ]  # Consume message and newline from buffer

                    message_str = message_bytes.decode("utf-8", errors="ignore").strip()
                    if not message_str:  # Skip if it was just an empty line (e.g. \n\n)
                        continue

                    self.logger.info(f"Received command string from Bus: {message_str}")
                    response_payload = {
                        "status": "ERROR",
                        "message": "Task processing failed",
                    }  # Default error response

                    try:
                        command_from_bus = json.loads(message_str)
                        task = command_from_bus.get("task", "UNKNOWN_TASK")
                        params = command_from_bus.get("params", {})

                        self.logger.info(
                            f"Processing task: '{task}' with params: {params}"
                        )

                        if task == "GET_STATUS":
                            self.telemetry_counter += 1
                            response_payload = {
                                "status": "SUCCESS",
                                "data": {
                                    "power": self.power_status,
                                    "payload_mode": self.payload_mode,
                                    "attitude": self.current_attitude,
                                    "software_version": "1.0.1",
                                    "id": self.software_id,
                                    "telemetry_count": self.telemetry_counter,
                                },
                            }
                        elif task == "SET_PAYLOAD_MODE":
                            new_mode = params.get("mode", self.payload_mode)
                            if new_mode in ["IDLE", "ACTIVE", "SAFE_MODE"]:
                                self.payload_mode = new_mode
                                response_payload = {
                                    "status": "SUCCESS",
                                    "message": f"Payload mode set to {new_mode}",
                                }
                                self.logger.info(f"Payload mode changed to: {new_mode}")
                            else:
                                response_payload = {
                                    "status": "ERROR",
                                    "message": f"Invalid payload mode: {new_mode}",
                                }
                        elif task == "SIMULATE_MANEUVER":
                            duration = params.get("duration_s", random.randint(1, 3))
                            self.logger.info(f"Simulating maneuver for {duration}s...")
                            await asyncio.sleep(duration)  # Simulate work
                            self.current_attitude["yaw"] = round(
                                self.current_attitude["yaw"] + random.uniform(-5, 5), 2
                            )
                            self.current_attitude["pitch"] = round(
                                self.current_attitude["pitch"] + random.uniform(-1, 1),
                                2,
                            )
                            response_payload = {
                                "status": "SUCCESS",
                                "message": f"Maneuver sim complete. Attitude: {self.current_attitude}",
                            }
                        else:
                            self.logger.warning(f"Unknown task received: {task}")
                            response_payload = {
                                "status": "ERROR",
                                "message": f"Unknown task: {task}",
                            }

                    except json.JSONDecodeError as jde:
                        self.logger.error(
                            f"JSON Decode Error for Bus command: {jde} - Data: '{message_str[:200]}'"
                        )
                        response_payload = {
                            "status": "ERROR",
                            "message": f"Invalid JSON command format: {jde}",
                        }
                    except Exception as e_proc:
                        self.logger.error(
                            f"Error processing Bus command task '{task}': {e_proc}",
                            exc_info=True,
                        )
                        response_payload = {
                            "status": "ERROR",
                            "message": f"Internal SW error processing task '{task}': {str(e_proc)}",
                        }

                    # Send response back to the Bus OS
                    if writer and not writer.is_closing():
                        try:
                            response_json_to_bus = json.dumps(response_payload) + "\n"
                            writer.write(response_json_to_bus.encode("utf-8"))
                            await writer.drain()
                            self.logger.info(
                                f"Sent response to Bus: {response_json_to_bus.strip()}"
                            )
                        except (ConnectionResetError, BrokenPipeError):
                            self.logger.warning(
                                f"Could not send response to Bus {peer_address_str}, connection lost."
                            )
                            # Raising an error here will break the outer loop and close connection
                            raise
                        except Exception as e_write:
                            self.logger.error(
                                f"Error sending response to Bus {peer_address_str}: {e_write}"
                            )
                            raise  # Propagate to break connection

        except asyncio.CancelledError:
            self.logger.info(f"Bus command handler for {peer_address_str} cancelled.")
        except ConnectionResetError:
            self.logger.info(f"Connection reset by Satellite Bus {peer_address_str}.")
        except asyncio.IncompleteReadError:
            self.logger.info(f"Incomplete read from Satellite Bus {peer_address_str}.")
        except Exception as e:
            self.logger.error(
                f"Error in Bus command handler with {peer_address_str}: {e}",
                exc_info=True,
            )
        finally:
            self.logger.info(
                f"Closing connection with Satellite Bus {peer_address_str}"
            )
            if writer and not writer.is_closing():
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception as e_c:
                    self.logger.error(f"Error during writer close for Bus: {e_c}")

    async def start_server(self):
        try:
            server = await asyncio.start_server(
                self.handle_bus_command, self.host, self.port
            )
            addr = (
                server.sockets[0].getsockname()
                if server.sockets
                else (self.host, self.port)
            )
            self.logger.info(
                f"Onboard Software TCP Server serving on {addr} for Satellite Bus."
            )
            async with server:
                await server.serve_forever()
        except OSError as e:
            self.logger.error(
                f"OSW Server: Could not start on {self.host}:{self.port}. Error: {e}"
            )
            raise
        except asyncio.CancelledError:
            self.logger.info("OSW Server task cancelled.")
        finally:
            self.logger.info("OSW Server shutting down.")


async def main_onboard_sw(args):
    onboard_sw = OnboardSoftware(software_id=args.id, host=args.host, port=args.port)
    try:
        await onboard_sw.start_server()
    except asyncio.CancelledError:
        onboard_sw.logger.info("OSW main task cancelled.")
    except OSError:
        pass  # Already logged by start_server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Onboard Software Service for a Satellite"
    )
    parser.add_argument(
        "--id", type=str, required=True, help="Unique Software ID (e.g., SatAlpha_SW)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host for this software to listen on",
    )
    parser.add_argument(
        "--port", type=int, required=True, help="Port for this software to listen on"
    )
    args = parser.parse_args()
    logger = logging.getLogger(f"OnboardSWMain[{args.id}]")
    try:
        logger.info(f"Starting Onboard Software Service {args.id}...")
        asyncio.run(main_onboard_sw(args))
    except KeyboardInterrupt:
        logger.info(f"OSW {args.id} stopped by user.")
    except Exception as e:
        logger.critical(f"Critical error in OSW {args.id}: {e}", exc_info=True)
    finally:
        logger.info(f"Application for OSW {args.id} shutting down.")
