# satellite_service.py
import asyncio
import json
import argparse  # For command line arguments


class Satellite:
    def __init__(self, satellite_id="SatDefault", host="localhost", port=65432):
        self.satellite_id = satellite_id
        self.host = host
        self.port = port
        self.current_state = {}
        print(
            f"[{self.satellite_id}] Initializing to listen on {self.host}:{self.port}..."
        )

    async def handle_client(self, reader, writer):
        addr = writer.get_extra_info("peername")
        print(f"[{self.satellite_id}] Accepted connection from {addr}")
        try:
            buffer = b""
            while True:
                data = await reader.read(4096)  # Read up to 4KB
                if not data:
                    if buffer:  # Process any remaining data in buffer before closing
                        print(
                            f"[{self.satellite_id}] Connection closing with remaining data in buffer: {buffer.decode(errors='ignore')}"
                        )
                    print(f"[{self.satellite_id}] Connection closed by {addr}")
                    break

                buffer += data

                # Process buffer for complete JSON messages
                while True:
                    try:
                        # Attempt to find a complete JSON message
                        # A simple way is to find the first '{' and matching '}'
                        # More robust: look for newline or other delimiters if your protocol defines one,
                        # or use a length-prefix for messages.
                        # For now, we'll assume one JSON object or a stream of them.
                        # This basic split assumes JSON objects are not nested in a way that confuses find('}')

                        # Let's try to decode messages separated by some delimiter or assume one message per packet for now
                        # For robust streaming JSON, libraries like `ijson` or a custom framing protocol would be better.
                        # This example will try to process JSON objects as they arrive.
                        # If multiple JSONs are sent in one packet without clear delimiters, this needs more robust parsing.

                        # Simplistic approach: try to decode the whole buffer.
                        # If it fails, wait for more data. If it succeeds, clear buffer or identify consumed part.
                        # This is prone to errors if multiple JSONs are in buffer without clear separation.
                        # For a more robust solution with potentially multiple JSONs in a stream:
                        message_str, consumed_len = self.extract_json_message(buffer)

                        if message_str:
                            message = json.loads(message_str)
                            if message.get("type") == "STATE_UPDATE":
                                self.current_state = message.get("payload", {})
                                print(
                                    f"[{self.satellite_id}] State Update @ SimTime {self.current_state.get('timestamp_sim_sec', 'N/A')}: "
                                    f"Pos: {self.current_state.get('position_eci_km', 'N/A')}, "
                                    f"Vel: {self.current_state.get('velocity_eci_km_s', 'N/A')}"
                                )
                            else:
                                print(
                                    f"[{self.satellite_id}] Received unknown message type: {message.get('type')}"
                                )
                            buffer = buffer[consumed_len:]  # Remove processed part
                            if not buffer:  # All buffer processed
                                break
                        else:  # No complete message found yet
                            if (
                                not data and buffer
                            ):  # No more data coming but buffer has partial message
                                print(
                                    f"[{self.satellite_id}] Partial message in buffer at connection close: {buffer.decode(errors='ignore')}"
                                )
                            break  # Wait for more data

                    except json.JSONDecodeError as jde:
                        # If buffer doesn't form a complete JSON yet and more data might come
                        if (
                            len(buffer) < 4096 and data
                        ):  # Arbitrary check: if buffer is not full and data was just read
                            # print(f"[{self.satellite_id}] Incomplete JSON, waiting for more data. Buffer: {buffer.decode(errors='ignore')}")
                            break
                        print(
                            f"[{self.satellite_id}] Error decoding JSON: {jde}. Buffer: {buffer.decode(errors='ignore')}"
                        )
                        # Clear buffer to prevent reprocessing invalid JSON, or implement more sophisticated error recovery
                        buffer = b""
                        break
                    except Exception as e:
                        print(f"[{self.satellite_id}] Error processing message: {e}")
                        buffer = b""  # Clear buffer on other errors
                        break

        except asyncio.CancelledError:
            print(f"[{self.satellite_id}] Client handler cancelled.")
        except ConnectionResetError:
            print(f"[{self.satellite_id}] Connection reset by {addr}.")
        except Exception as e:
            print(f"[{self.satellite_id}] Connection error with {addr}: {e}")
        finally:
            print(f"[{self.satellite_id}] Closing connection with {addr}")
            if writer and not writer.is_closing():
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception as e_close:
                    print(
                        f"[{self.satellite_id}] Error during writer close for {addr}: {e_close}"
                    )

    def extract_json_message(self, buffer_bytes):
        """
        Tries to extract the first complete JSON message from the buffer.
        Returns (message_str, consumed_length) or (None, 0)
        This is a simplified implementation. A robust solution would handle nested
        structures and edge cases more gracefully, or use a length-prefix framing.
        """
        buffer_str = buffer_bytes.decode(errors="ignore")
        open_braces = 0
        start_index = -1

        for i, char in enumerate(buffer_str):
            if char == "{":
                if start_index == -1:
                    start_index = i
                open_braces += 1
            elif char == "}":
                if start_index != -1:  # Ensure we are inside a potential JSON object
                    open_braces -= 1
                    if open_braces == 0:
                        # Found a potential complete JSON object
                        json_str = buffer_str[start_index : i + 1]
                        try:
                            json.loads(json_str)  # Validate if it's actually JSON
                            return json_str, len(
                                json_str.encode()
                            )  # Return validated string and its byte length
                        except json.JSONDecodeError:
                            # Invalid JSON, maybe part of a larger message or corrupt data.
                            # Continue searching or implement error handling.
                            # For this example, we'll assume it's not a valid delimiter point
                            # and continue searching by resetting start_index if we want to find NEXT valid JSON.
                            # However, if a non-valid JSON is found, it might be better to flag error.
                            # Resetting state to find next valid JSON:
                            # open_braces = 0
                            # start_index = -1
                            # For simplicity, we'll just let the outer loop handle JSONDecodeError for now.
                            # This function aims to find a *structurally* complete object.
                            pass  # Let the json.loads in handle_client do the main validation. This is just for finding boundaries.

        return None, 0  # No complete message found

    async def start_server(self):
        server = await asyncio.start_server(self.handle_client, self.host, self.port)

        addr = server.sockets[0].getsockname()
        print(f"[{self.satellite_id}] Serving on {addr}")

        async with server:
            try:
                await server.serve_forever()
            except asyncio.CancelledError:
                print(f"[{self.satellite_id}] Server task cancelled.")
            finally:
                print(f"[{self.satellite_id}] Server shutting down.")
                server.close()
                await server.wait_closed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Satellite Service")
    parser.add_argument(
        "--id", type=str, default="SatDefault", help="Unique Satellite ID"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to listen on (e.g., 0.0.0.0 for all interfaces)",
    )
    parser.add_argument("--port", type=int, default=65432, help="Port to listen on")
    args = parser.parse_args()

    # This instantiation should only happen ONCE.
    # The repeated "Initializing..." message in your output is strange.
    # Please ensure this line (and the print within Satellite.__init__)
    # is not accidentally inside a loop in your local copy of the script.
    satellite = Satellite(satellite_id=args.id, host=args.host, port=args.port)

    try:
        # Use asyncio.run() to manage the event loop and run the main coroutine
        asyncio.run(satellite.start_server())
    except KeyboardInterrupt:
        print(
            f"\n[{satellite.satellite_id}] Server stopped by user (KeyboardInterrupt)."
        )
    except OSError as e:
        # This is typically for issues like "port already in use"
        print(
            f"[{satellite.satellite_id}] ERROR: Could not start server on {args.host}:{args.port}. Details: {e}"
        )
    except Exception as e_main:
        # Catch any other unexpected errors during asyncio.run or server setup
        print(f"[{satellite.satellite_id}] An unexpected error occurred: {e_main}")
    finally:
        # asyncio.run() handles loop cleanup, so this can be simpler.
        print(f"[{satellite.satellite_id}] Application for {args.id} is shutting down.")
