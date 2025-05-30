# satellite_service.py (Bus OS)
import asyncio
import json
import argparse
import logging
import subprocess  # For launching the onboard software process
import sys  # To get PYTHON_EXECUTABLE
import os  # For path operations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s][%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
PYTHON_EXECUTABLE = sys.executable


class Satellite:  # This is now acting as the "Bus OS"
    def __init__(
        self,
        satellite_id="SatDefault",
        host="localhost",
        port=65432,
        onboard_sw_script_name="onboard_software_service.py",
    ):
        self.satellite_id = satellite_id
        self.host = host  # Host this Bus OS listens on for Core/GS
        self.port = port  # Port this Bus OS listens on
        self.current_state_from_core = {}
        self.logger = logging.getLogger(f"SatelliteBus[{self.satellite_id}]")

        # Determine script path relative to this script's location
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.onboard_sw_script_path = os.path.join(
            current_script_dir, onboard_sw_script_name
        )

        self.onboard_sw_comm_details = {
            "host": "localhost",
            "port": self.port + 1000,  # OSW listens on BusPort + 1000
            "id": f"{self.satellite_id}_OSW",  # Onboard Software ID
        }
        self.onboard_software_process: asyncio.subprocess.Process | None = None
        self.onboard_sw_stdout_task = None
        self.onboard_sw_stderr_task = None
        self._main_server_obj = None  # To store the asyncio.Server object

        self.logger.info(
            f"Initializing. Bus OS will listen on {self.host}:{self.port} for Core/GS."
        )
        self.logger.info(
            f"Configured to launch Onboard SW '{self.onboard_sw_script_path}' which should listen on "
            f"{self.onboard_sw_comm_details['host']}:{self.onboard_sw_comm_details['port']}"
        )

    async def _launch_and_monitor_onboard_software(self):
        command = [
            PYTHON_EXECUTABLE,
            "-u",
            self.onboard_sw_script_path,
            "--id",
            self.onboard_sw_comm_details["id"],
            "--host",
            self.onboard_sw_comm_details["host"],
            "--port",
            str(self.onboard_sw_comm_details["port"]),
        ]
        self.logger.info(f"Attempting to launch OSW: {' '.join(command)}")
        try:
            self.onboard_software_process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            self.logger.info(
                f"OSW for {self.satellite_id} launched (PID: {self.onboard_software_process.pid})."
            )
            self.onboard_sw_stdout_task = asyncio.create_task(
                self._stream_subprocess_output(
                    self.onboard_software_process.stdout, "OSW_OUT"
                )
            )
            self.onboard_sw_stderr_task = asyncio.create_task(
                self._stream_subprocess_output(
                    self.onboard_software_process.stderr, "OSW_ERR"
                )
            )
            # Optionally, monitor for unexpected exit:
            asyncio.create_task(self._monitor_osw_exit())
        except FileNotFoundError:
            self.logger.error(
                f"OSW script '{self.onboard_sw_script_path}' not found! Bus cannot function."
            )
            self.onboard_software_process = None
            raise  # Critical failure
        except Exception as e:
            self.logger.error(f"Failed to launch OSW: {e}", exc_info=True)
            self.onboard_software_process = None
            raise  # Critical failure

    async def _monitor_osw_exit(self):
        if self.onboard_software_process:
            return_code = await self.onboard_software_process.wait()
            self.logger.warning(
                f"Onboard Software process for {self.satellite_id} exited with code {return_code}."
            )
            # Here you could implement restart logic if desired.
            self.onboard_software_process = None  # Mark as no longer running

    async def _stream_subprocess_output(
        self, stream: asyncio.StreamReader, prefix: str
    ):
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                self.logger.info(
                    f"[{prefix}/{self.satellite_id}] {line.decode(errors='ignore').strip()}"
                )
        except asyncio.CancelledError:
            self.logger.info(
                f"Streaming task for {prefix}/{self.satellite_id} cancelled."
            )
        except Exception as e:
            self.logger.error(f"Error streaming {prefix} for {self.satellite_id}: {e}")
        finally:
            self.logger.info(f"Streaming for {prefix}/{self.satellite_id} finished.")

    async def _send_command_to_onboard_sw(self, command_payload_dict: dict):
        if (
            not self.onboard_software_process
            or self.onboard_software_process.returncode is not None
        ):
            self.logger.error("OSW process not running. Cannot send command.")
            return {"status": "ERROR", "message": "Onboard software not running"}

        osw_host, osw_port = (
            self.onboard_sw_comm_details["host"],
            self.onboard_sw_comm_details["port"],
        )
        reader, writer = None, None
        try:
            self.logger.debug(
                f"Bus connecting to OSW at {osw_host}:{osw_port} for command: {command_payload_dict}"
            )
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(osw_host, osw_port), timeout=3.0
            )
            command_json = json.dumps(command_payload_dict) + "\n"
            writer.write(command_json.encode("utf-8"))
            await writer.drain()
            self.logger.debug("Command sent to OSW, awaiting response...")
            response_data = await asyncio.wait_for(
                reader.readuntil(b"\n"), timeout=10.0
            )
            response_dict = json.loads(response_data.decode("utf-8").strip())
            self.logger.debug(f"Response from OSW: {response_dict}")
            return response_dict
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout with OSW at {osw_host}:{osw_port}.")
            return {"status": "ERROR", "message": "Timeout with OSW"}
        except ConnectionRefusedError:
            self.logger.error(f"OSW connection refused at {osw_host}:{osw_port}.")
            return {"status": "ERROR", "message": "OSW connection refused"}
        except json.JSONDecodeError:
            self.logger.error("Failed to decode JSON from OSW.")
            return {"status": "ERROR", "message": "Invalid JSON response from OSW"}
        except Exception as e:
            self.logger.error(f"Error sending to OSW: {e}", exc_info=True)
            return {"status": "ERROR", "message": f"IPC error: {str(e)}"}
        finally:
            if writer and not writer.is_closing():
                try:
                    writer.close()
                    await writer.wait_closed()
                except:
                    pass

    def _parse_json_message(self, data_bytes: bytes) -> tuple[dict | None, bytes]:
        try:
            newline_index = data_bytes.find(b"\n")
            if newline_index != -1:
                json_str = data_bytes[:newline_index].decode("utf-8", errors="ignore")
                buffer = data_bytes[newline_index + 1 :]
                return json.loads(json_str), buffer
        except json.JSONDecodeError as jde:
            self.logger.error(
                f"JSON Decode Error (Bus): {jde} for data: '{data_bytes[:200].decode(errors='ignore')}...'"
            )
            if newline_index != -1:
                return None, data_bytes[newline_index + 1 :]  # Discard corrupt
            return None, data_bytes
        except Exception as e:
            self.logger.error(f"Error parsing (Bus): {e}")
            if newline_index != -1:
                return None, data_bytes[newline_index + 1 :]
            return None, data_bytes
        return None, data_bytes

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        peer_name = writer.get_extra_info("peername", "UnknownPeer")
        self.logger.info(f"Accepted connection from {peer_name}")
        buffer = b""
        try:
            while True:
                data_chunk = await reader.read(4096)
                if not data_chunk:
                    self.logger.info(f"Connection from {peer_name} closed (EOF).")
                    break
                buffer += data_chunk
                while True:
                    message_dict, remaining_buffer = self._parse_json_message(buffer)
                    buffer = remaining_buffer
                    if message_dict:
                        self.logger.debug(f"From {peer_name}: {message_dict}")
                        msg_type, payload = message_dict.get("type"), message_dict.get(
                            "payload"
                        )
                        resp_str = None
                        if msg_type == "STATE_UPDATE":
                            self.current_state_from_core = payload or {}
                            # self.logger.info(f"Core State Update @ {self.current_state_from_core.get('timestamp_sim_sec', 'N/A')}")
                        elif msg_type == "GROUND_COMMAND":
                            self.logger.info(
                                f"GROUND_COMMAND from {peer_name} (payload: {payload}). Forwarding to OSW."
                            )
                            if isinstance(payload, dict):
                                osw_resp = await self._send_command_to_onboard_sw(
                                    payload
                                )
                                resp_str = json.dumps(
                                    {
                                        "type": "GROUND_COMMAND_ACK",
                                        "sat_id": self.satellite_id,
                                        "osw_response": osw_resp,
                                    }
                                )
                            else:
                                resp_str = json.dumps(
                                    {
                                        "type": "ERROR",
                                        "message": "Invalid GROUND_COMMAND payload",
                                    }
                                )
                        else:
                            self.logger.warning(
                                f"Unknown message type '{msg_type}' from {peer_name}"
                            )
                            resp_str = json.dumps(
                                {
                                    "type": "ERROR",
                                    "message": f"Unknown message type '{msg_type}'",
                                }
                            )

                        if (
                            resp_str and writer and not writer.is_closing()
                        ):  # Only respond if there's something to say (e.g. to GS)
                            try:
                                writer.write((resp_str + "\n").encode("utf-8"))
                                await writer.drain()
                                self.logger.info(
                                    f"Sent to {peer_name}: {resp_str[:100]}..."
                                )
                            except Exception as e_w:
                                self.logger.warning(
                                    f"Failed to send to {peer_name}: {e_w}"
                                )
                                break
                    else:
                        break  # No complete message in buffer
        except (
            asyncio.CancelledError,
            ConnectionResetError,
            asyncio.IncompleteReadError,
        ):
            pass  # Expected
        except Exception as e:
            self.logger.error(f"Error with {peer_name}: {e}", exc_info=True)
        finally:
            self.logger.info(f"Closing connection with {peer_name}")
            if writer and not writer.is_closing():
                try:
                    writer.close()
                    await writer.wait_closed()
                except:
                    pass

    async def start_all_services(self):
        self.logger.info("Attempting to launch dependent Onboard Software...")
        try:
            await self._launch_and_monitor_onboard_software()
        except (
            Exception
        ) as e_osw_launch:  # Catch launch failure from _launch_and_monitor_onboard_software
            self.logger.critical(
                f"Onboard Software failed to launch: {e_osw_launch}. Satellite Bus OS will not start main server."
            )
            return  # Do not proceed if OSW fails

        if (
            not self.onboard_software_process
            or self.onboard_software_process.returncode is not None
        ):
            self.logger.critical(
                "Onboard Software launch confirmation failed. Bus OS server not starting."
            )
            return

        self.logger.info("Onboard Software launched. Starting main Bus OS server...")
        try:
            server = await asyncio.start_server(
                self.handle_client, self.host, self.port
            )
            self._main_server_obj = server  # Store for graceful shutdown
            addr = (
                server.sockets[0].getsockname()
                if server.sockets
                else (self.host, self.port)
            )
            self.logger.info(f"Satellite Bus OS Server serving on {addr} for Core/GS.")
            async with server:
                await server.serve_forever()
        except OSError as e:
            self.logger.error(
                f"Bus OS Server: Could not start on {self.host}:{self.port}. Error: {e}"
            )
            raise
        except asyncio.CancelledError:
            self.logger.info("Bus OS server task cancelled.")
        finally:
            self.logger.info("Bus OS server shutting down.")

    async def stop_all_services(self):
        self.logger.info(
            f"Satellite Bus OS [{self.satellite_id}] initiating shutdown of all services..."
        )
        if (
            hasattr(self, "_main_server_obj")
            and self._main_server_obj
            and self._main_server_obj.is_serving()
        ):
            self._main_server_obj.close()
            await self._main_server_obj.wait_closed()
            self.logger.info("Main Bus OS TCP server stopped.")

        tasks_to_cancel_osw = []
        if (
            hasattr(self, "onboard_sw_stdout_task")
            and self.onboard_sw_stdout_task
            and not self.onboard_sw_stdout_task.done()
        ):
            tasks_to_cancel_osw.append(self.onboard_sw_stdout_task)
        if (
            hasattr(self, "onboard_sw_stderr_task")
            and self.onboard_sw_stderr_task
            and not self.onboard_sw_stderr_task.done()
        ):
            tasks_to_cancel_osw.append(self.onboard_sw_stderr_task)
        for task in tasks_to_cancel_osw:
            task.cancel()

        if (
            hasattr(self, "onboard_software_process")
            and self.onboard_software_process
            and self.onboard_software_process.returncode is None
        ):
            self.logger.info(
                f"Terminating Onboard Software process (PID: {self.onboard_software_process.pid})..."
            )
            try:
                self.onboard_software_process.terminate()
                await asyncio.wait_for(
                    self.onboard_software_process.wait(), timeout=5.0
                )
                self.logger.info("OSW process terminated.")
            except asyncio.TimeoutError:
                self.logger.warning("OSW process timeout, killing...")
                self.onboard_software_process.kill()
            except ProcessLookupError:
                self.logger.info("OSW process already gone.")
            except Exception as e:
                self.logger.error(f"Error terminating OSW: {e}")

        if tasks_to_cancel_osw:
            try:
                await asyncio.gather(*tasks_to_cancel_osw, return_exceptions=True)
                self.logger.info("OSW log streaming tasks finalized.")
            except Exception as e_g:
                self.logger.error(f"Error gathering OSW log tasks: {e_g}")
        self.logger.info(f"Satellite {self.satellite_id} services fully stopped.")


async def main_satellite_bus(args):
    satellite_bus = Satellite(
        satellite_id=args.id,
        host=args.host,
        port=args.port,
        onboard_sw_script_name=args.osw_script,
    )
    try:
        await satellite_bus.start_all_services()
    except asyncio.CancelledError:
        satellite_bus.logger.info("Satellite Bus main runner cancelled.")
    except Exception as e:
        satellite_bus.logger.critical(
            f"Bus failed to start/run services: {e}", exc_info=True
        )
    finally:
        await satellite_bus.stop_all_services()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Satellite Bus OS Service with Onboard Software Subprocess"
    )
    parser.add_argument(
        "--id",
        type=str,
        default="SatDefault",
        help="Unique Satellite ID for this Bus OS",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host for this Bus OS to listen on",
    )
    parser.add_argument(
        "--port", type=int, default=65432, help="Port for this Bus OS to listen on"
    )
    parser.add_argument(
        "--osw-script",
        type=str,
        default="onboard_software_service.py",
        help="Path to the onboard software script",
    )
    args = parser.parse_args()
    main_logger = logging.getLogger(f"SatelliteBusMain[{args.id}]")
    try:
        main_logger.info(f"Starting Satellite Bus OS for {args.id}...")
        asyncio.run(main_satellite_bus(args))
    except KeyboardInterrupt:
        main_logger.info(f"Bus OS for {args.id} stopped by user.")
    except Exception as e:
        main_logger.critical(
            f"Critical error in Bus OS {args.id} main: {e}", exc_info=True
        )
    finally:
        main_logger.info(f"Application for Satellite Bus OS {args.id} shutting down.")
