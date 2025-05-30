import asyncio
from src.core.simulation_core import SimulationCore


async def core_main_runner(
    sim_control_port,
    sim_vis_port,
    sim_gs_update_port,
):
    core = SimulationCore(
        control_port=sim_control_port,
        vis_port=sim_vis_port,
        gs_update_port=sim_gs_update_port,
    )
    servers = []
    server_tasks = []

    try:
        control_server = await asyncio.start_server(
            core.handle_control_command, "localhost", core.control_port
        )
        servers.append(control_server)
        addr = control_server.sockets[0].getsockname()
        print(f"[Core] Control interface listening on {addr}")
        server_tasks.append(
            asyncio.create_task(control_server.serve_forever(), name="ControlServer")
        )

        vis_server = await asyncio.start_server(
            core.handle_visualizer_connection, "localhost", core.vis_port
        )
        servers.append(vis_server)
        addr = vis_server.sockets[0].getsockname()
        print(f"[Core] Visualization interface listening on {addr}")
        server_tasks.append(
            asyncio.create_task(vis_server.serve_forever(), name="VisServer")
        )

        gs_update_server = await asyncio.start_server(
            core.handle_groundstation_connection, "localhost", core.gs_update_port
        )
        servers.append(gs_update_server)
        addr = gs_update_server.sockets[0].getsockname()
        print(f"[Core] Groundstation Update interface listening on {addr}")
        server_tasks.append(
            asyncio.create_task(gs_update_server.serve_forever(), name="GSUpdateServer")
        )

        simulation_task = asyncio.create_task(
            core.run_simulation_loop(), name="SimulationLoop"
        )
        await simulation_task
    except OSError as e:
        print(
            f"[Core] FATAL: Could not start a server: {e}. Ensure ports are free. Exiting."
        )
    except asyncio.CancelledError:
        print("[Core] Main runner task cancelled.")
    except Exception as e_main_run:
        print(f"[Core] Exception in core_main_runner: {e_main_run}")
    finally:
        print("[Core] Shutting down server tasks and interfaces...")
        for task in server_tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass  # Expected
                except Exception as e_task_cancel:
                    print(
                        f"[Core] Error cancelling server task {task.get_name()}: {e_task_cancel}"
                    )

        for server in servers:
            if server and server.is_serving():
                server.close()
                try:
                    await server.wait_closed()
                except Exception as e_server_close:
                    print(f"[Core] Error closing server: {e_server_close}")
        print("[Core] All server interfaces shut down.")


if __name__ == "__main__":
    initial_groundstation_configurations = [
        {"id": "GS_London", "lat_deg": 51.5074, "lon_deg": -0.1278, "alt_km": 0.035},
        {"id": "GS_NewYork", "lat_deg": 40.7128, "lon_deg": -74.0060, "alt_km": 0.010},
        {"id": "GS_Sydney", "lat_deg": -33.8688, "lon_deg": 151.2093, "alt_km": 0.058},
    ]

    CORE_CONTROL_PORT = 60000
    CORE_VIS_PORT = 60001
    CORE_GS_UPDATE_PORT = 60002

    try:
        asyncio.run(
            core_main_runner(
                CORE_CONTROL_PORT,
                CORE_VIS_PORT,
                CORE_GS_UPDATE_PORT,
            )
        )
    except KeyboardInterrupt:
        print("\n[Core] Simulation Core stopped by user (KeyboardInterrupt).")
    except Exception as e_global:
        print(f"[Core] A critical error occurred in the main execution: {e_global}")
    finally:
        print("[Core] Application shutdown complete.")
