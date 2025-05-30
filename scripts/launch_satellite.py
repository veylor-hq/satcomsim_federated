import asyncio
import argparse

from src.satellite.satellite_service import Satellite


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
    satellite = Satellite(satellite_id=args.id, host=args.host, port=args.port)

    try:
        asyncio.run(satellite.start_server())
    except KeyboardInterrupt:
        print(
            f"\n[{satellite.satellite_id}] Server stopped by user (KeyboardInterrupt)."
        )
    except OSError as e:
        print(
            f"[{satellite.satellite_id}] ERROR: Could not start server on {args.host}:{args.port}. Details: {e}"
        )
    except Exception as e_main:
        print(f"[{satellite.satellite_id}] An unexpected error occurred: {e_main}")
    finally:
        print(f"[{satellite.satellite_id}] Application for {args.id} is shutting down.")
