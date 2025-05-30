import asyncio
import json


class GroundstationService:
    def __init__(self, gs_id, core_host, core_port, user_listen_host, user_listen_port):
        self.gs_id = gs_id
        self.core_host = core_host
        self.core_port = core_port
        self.user_listen_host = user_listen_host
        self.user_listen_port = user_listen_port

        self.current_eci_pos_km = None
        self.visible_satellites_info = (
            {}
        )  # {sat_id: {"connect_host": host, "connect_port": port}}
        self.lock = asyncio.Lock()

        self._core_connection_task = None
        self._user_command_server = None  # asyncio.Server object
        self._user_server_runner_task = None  # Task for serve_forever()

    async def _connect_to_core_and_listen(self):
        while True:
            reader, writer = None, None  # Ensure they are defined for finally block
            try:
                print(
                    f"[{self.gs_id}] Attempting to connect to Simulation Core GS Update Stream at {self.core_host}:{self.core_port}..."
                )
                reader, writer = await asyncio.open_connection(
                    self.core_host, self.core_port
                )
                print(f"[{self.gs_id}] Connected to Simulation Core GS Update Stream.")

                buffer = ""
                while True:
                    data = await reader.read(8192)
                    if not data:
                        print(f"[{self.gs_id}] Connection to Core lost (EOF).")
                        break

                    buffer += data.decode("utf-8", errors="ignore")

                    while "\n" in buffer:
                        message_str, buffer = buffer.split("\n", 1)
                        if not message_str.strip():
                            continue

                        try:
                            message = json.loads(message_str)
                            if message.get("type") == "GS_SIM_UPDATE":
                                async with self.lock:
                                    self.visible_satellites_info.clear()
                                    found_self = False
                                    for gs_data_entry in message.get(
                                        "groundstations_data", []
                                    ):
                                        if gs_data_entry.get("id") == self.gs_id:
                                            found_self = True
                                            self.current_eci_pos_km = gs_data_entry.get(
                                                "eci_pos_km"
                                            )
                                            for sat_info in gs_data_entry.get(
                                                "visible_sats", []
                                            ):
                                                self.visible_satellites_info[
                                                    sat_info["id"]
                                                ] = {
                                                    "connect_host": sat_info[
                                                        "connect_host"
                                                    ],
                                                    "connect_port": sat_info[
                                                        "connect_port"
                                                    ],
                                                }
                                            break
                                    if not found_self:
                                        if (
                                            self.current_eci_pos_km
                                            or self.visible_satellites_info
                                        ):
                                            print(
                                                f"[{self.gs_id}] Did not find own ID in core update. Clearing visibility."
                                            )
                                            self.current_eci_pos_km = None
                                            self.visible_satellites_info.clear()
                        except json.JSONDecodeError as jde:
                            print(
                                f"[{self.gs_id}] JSON Decode Error from Core: {jde} for '{message_str[:100]}...'"
                            )
                        except Exception as e:
                            print(
                                f"[{self.gs_id}] Error processing message from Core: {e}"
                            )

            except ConnectionRefusedError:
                print(
                    f"[{self.gs_id}] Connection to Core refused. Retrying in 10 seconds..."
                )
            except (
                OSError,
                asyncio.TimeoutError,
                asyncio.IncompleteReadError,
                ConnectionResetError,
            ) as e:
                print(
                    f"[{self.gs_id}] Connection to Core failed or lost ({type(e).__name__}). Retrying in 10s..."
                )
            except asyncio.CancelledError:
                print(f"[{self.gs_id}] Core connection task cancelled.")
                break  # Exit while True loop if cancelled
            except Exception as e_outer:
                print(
                    f"[{self.gs_id}] Unexpected error in Core connection loop: {e_outer}. Retrying in 10s."
                )
            finally:
                if writer and not writer.is_closing():
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except:
                        pass  # Ignore errors during close
                async with self.lock:
                    self.current_eci_pos_km = None
                    self.visible_satellites_info.clear()

            if (
                not asyncio.current_task().done()
            ):  # Check if task was cancelled before sleeping
                await asyncio.sleep(10)

    async def send_command_to_satellite(self, target_sat_id, payload_str):
        connect_info = None
        async with self.lock:
            connect_info = self.visible_satellites_info.get(target_sat_id)

        if not connect_info:
            msg = f"Satellite {target_sat_id} is not currently visible or known."
            print(f"[{self.gs_id}] {msg}")
            return f"ERROR: {msg}"

        sat_host, sat_port = connect_info["connect_host"], connect_info["connect_port"]
        reader_sat, writer_sat = None, None  # Ensure defined for finally
        try:
            print(
                f"[{self.gs_id}] Attempting to connect to satellite {target_sat_id} at {sat_host}:{sat_port}..."
            )
            reader_sat, writer_sat = await asyncio.wait_for(
                asyncio.open_connection(sat_host, sat_port), timeout=5.0
            )
            print(
                f"[{self.gs_id}] Connected to {target_sat_id}. Sending: {payload_str[:50]}..."
            )

            writer_sat.write(payload_str.encode("utf-8") + b"\n")
            await writer_sat.drain()

            response_data = await asyncio.wait_for(
                reader_sat.readuntil(b"\n"), timeout=10.0
            )
            response_str = response_data.decode("utf-8").strip()
            print(f"[{self.gs_id}] Response from {target_sat_id}: {response_str}")
            return f"OK: Response from {target_sat_id}: {response_str}"
        except asyncio.TimeoutError:
            msg = f"Timeout communicating with satellite {target_sat_id}."
            print(f"[{self.gs_id}] {msg}")
            return f"ERROR: {msg}"
        except ConnectionRefusedError:
            msg = f"Connection refused by satellite {target_sat_id}."
            print(f"[{self.gs_id}] {msg}")
            return f"ERROR: {msg}"
        except Exception as e:
            msg = f"Error with satellite {target_sat_id}: {e}"
            print(f"[{self.gs_id}] {msg}")
            return f"ERROR: {msg}"
        finally:
            if writer_sat and not writer_sat.is_closing():
                try:
                    writer_sat.close()
                    await writer_sat.wait_closed()
                except:
                    pass

    async def handle_user_command_connection(self, reader, writer):
        peer_name = writer.get_extra_info("peername")
        print(f"[{self.gs_id}] User connection from {peer_name}")
        try:
            while True:
                command_bytes = await reader.readuntil(b"\n")
                if not command_bytes:
                    break
                command_str = command_bytes.decode("utf-8").strip()
                if not command_str:
                    continue
                print(f"[{self.gs_id}] User command: {command_str}")
                parts = command_str.split(" ", 2)
                response_to_user = "ERROR: Unknown command. Use: SEND <sat_id> <payload> | LIST_VISIBLE | QUIT\n"

                if len(parts) >= 1 and parts[0].upper() == "SEND":
                    if len(parts) == 3:
                        target_sat_id, payload = parts[1], parts[2]
                        response_to_user = (
                            await self.send_command_to_satellite(target_sat_id, payload)
                            + "\n"
                        )
                    else:
                        response_to_user = (
                            "ERROR: SEND command format: SEND <sat_id> <payload>\n"
                        )
                elif command_str.upper() == "LIST_VISIBLE":
                    async with self.lock:
                        vis_sats = list(self.visible_satellites_info.keys())
                    response_to_user = (
                        f"OK: Visible: {', '.join(vis_sats) if vis_sats else 'None'}\n"
                    )
                elif command_str.upper() == "QUIT":
                    response_to_user = "OK: Closing connection.\n"
                    writer.write(response_to_user.encode("utf-8"))
                    await writer.drain()
                    break

                writer.write(response_to_user.encode("utf-8"))
                await writer.drain()
        except asyncio.IncompleteReadError:
            print(f"[{self.gs_id}] User {peer_name} disconnected (incomplete).")
        except ConnectionResetError:
            print(f"[{self.gs_id}] User {peer_name} reset connection.")
        except asyncio.CancelledError:
            print(f"[{self.gs_id}] User handler for {peer_name} cancelled.")
        except Exception as e:
            print(f"[{self.gs_id}] Error with user {peer_name}: {e}")
        finally:
            print(f"[{self.gs_id}] Closing user connection with {peer_name}")
            if writer and not writer.is_closing():
                try:
                    writer.close()
                    await writer.wait_closed()
                except:
                    pass

    async def start_services(self):
        """Starts and manages the core connection and user command server."""
        self._core_connection_task = asyncio.create_task(
            self._connect_to_core_and_listen(), name=f"{self.gs_id}_CoreLink"
        )

        self._user_command_server = await asyncio.start_server(
            self.handle_user_command_connection,
            self.user_listen_host,
            self.user_listen_port,
        )
        addr = self._user_command_server.sockets[0].getsockname()
        print(f"[{self.gs_id}] Now listening for user commands on {addr}")

        self._user_server_runner_task = asyncio.create_task(
            self._user_command_server.serve_forever(),
            name=f"{self.gs_id}_UserServerRunner",
        )

        # Keep running until one of the main tasks exits or is cancelled
        # Typically, this will run until KeyboardInterrupt cancels asyncio.run()
        done, pending = await asyncio.wait(
            [self._core_connection_task, self._user_server_runner_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # If one task finishes (e.g. _core_connection_task if it hits a break), cancel the other
        for task in pending:
            task.cancel()

        # Await all pending tasks to allow them to handle cancellation
        if pending:
            await asyncio.wait(pending)

        # Check for exceptions in completed tasks
        for task in done:
            try:
                task.result()  # This will raise an exception if the task failed
            except asyncio.CancelledError:
                print(f"[{self.gs_id}] Task {task.get_name()} was cancelled.")
            except Exception as e:
                print(f"[{self.gs_id}] Task {task.get_name()} finished with error: {e}")

    async def stop_services(self):
        """Gracefully stops all running services."""
        print(f"[{self.gs_id}] Initiating shutdown of services...")

        tasks_to_cancel = []
        if (
            hasattr(self, "_user_server_runner_task")
            and self._user_server_runner_task
            and not self._user_server_runner_task.done()
        ):
            tasks_to_cancel.append(self._user_server_runner_task)
        if (
            hasattr(self, "_core_connection_task")
            and self._core_connection_task
            and not self._core_connection_task.done()
        ):
            tasks_to_cancel.append(self._core_connection_task)

        for task in tasks_to_cancel:
            task.cancel()

        if self._user_command_server and self._user_command_server.is_serving():
            self._user_command_server.close()
            # Not awaiting wait_closed here because serve_forever task needs to exit first
            print(f"[{self.gs_id}] User command server close initiated.")

        # Await cancellation of tasks
        for task in tasks_to_cancel:
            try:
                await task
            except asyncio.CancelledError:
                print(f"[{self.gs_id}] Task {task.get_name()} confirmed cancelled.")
            except Exception as e:  # Should ideally not happen if cancellation is clean
                print(
                    f"[{self.gs_id}] Error awaiting cancelled task {task.get_name()}: {e}"
                )

        if self._user_command_server:  # Now ensure wait_closed if it was serving
            try:
                if (
                    self._user_command_server.is_serving()
                ):  # Check again, might have closed due to task cancel
                    pass  # Server is closed, task cancellation handles serve_forever exit
                # await self._user_command_server.wait_closed() # This can hang if serve_forever wasn't awaited after cancellation
                print(f"[{self.gs_id}] User command server fully stopped.")
            except Exception as e:
                print(
                    f"[{self.gs_id}] Error during user command server wait_closed: {e}"
                )

        print(f"[{self.gs_id}] Groundstation Service fully stopped.")
