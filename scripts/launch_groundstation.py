import asyncio
import argparse
from src.groundstation.groundstation_service import GroundstationService


async def main_groundstation_runner(gs_service_instance):
    """Main runner for the groundstation service that handles startup and shutdown."""
    try:
        await gs_service_instance.start_services()
    except asyncio.CancelledError:
        print(
            f"[{gs_service_instance.gs_id}] Main runner cancelled. Initiating stop..."
        )
    except Exception as e:
        print(f"[{gs_service_instance.gs_id}] Main runner encountered an error: {e}")
    finally:
        print(f"[{gs_service_instance.gs_id}] Main runner stopping services...")
        await gs_service_instance.stop_services()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Groundstation Service for SatComSim")
    parser.add_argument(
        "--id",
        type=str,
        required=True,
        help="Unique Groundstation ID (e.g., GS_London)",
    )
    parser.add_argument(
        "--core-host", type=str, default="localhost", help="Simulation Core host"
    )
    parser.add_argument(
        "--core-port",
        type=int,
        default=60002,
        help="Simulation Core's GS Update Stream port",
    )
    parser.add_argument(
        "--listen-host",
        type=str,
        default="localhost",
        help="Host for this GS to listen for user commands",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        required=True,
        help="Port for this GS to listen for user commands (e.g., 7000)",
    )

    args = parser.parse_args()

    gs_service = GroundstationService(
        gs_id=args.id,
        core_host=args.core_host,
        core_port=args.core_port,
        user_listen_host=args.listen_host,
        user_listen_port=args.listen_port,
    )

    try:
        print(f"[{args.id}] Starting groundstation service. Press Ctrl+C to stop.")
        asyncio.run(main_groundstation_runner(gs_service))
    except KeyboardInterrupt:
        print(f"\n[{args.id}] KeyboardInterrupt received by script. Shutting down...")
    except Exception as e_main_script:
        print(f"[{args.id}] Critical error in script __main__: {e_main_script}")
    finally:
        print(f"[{args.id}] Groundstation application script finished.")
