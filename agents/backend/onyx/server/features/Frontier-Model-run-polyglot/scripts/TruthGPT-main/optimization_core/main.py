"""
🚀 TruthGPT Command Center — Modular Orchestrator
System 5.9 Gold Standard
"""
import sys
import asyncio
from pathlib import Path
from rich.prompt import Prompt

# --- Path Initialization ---
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Modular Imports
from interface.core import (
    console, clear_screen, linux_boot_sequence, show_main_dashboard, 
    handle_personalize
)
from interface.swarm_menu import swarm_menu, handle_swarm_ask
from interface.model_menu import models_menu
from interface.research_menu import research_menu, intelligence_labs_menu
from interface.system_menu import system_menu, opts_menu, kernel_menu
from interface.blockchain_menu import blockchain_menu
from interface.comm_menu import handle_messaging_apps, marketing_intelligence_menu, embodied_rl_menu
from interface.infra_menu import infrastructure_menu, task_registry_menu
from interface.overdrive_menu import handle_overdrive_menu

async def main_loop():
    linux_boot_sequence()
    extended_mode = True
    while True:

        await show_main_dashboard(extended=extended_mode)
        
        user_input = Prompt.ask("[bold green]truthgpt@kernel[/bold green]:[bold blue]~[/bold blue]#", default="0")
        
        if user_input == "99" or user_input == "+":
            extended_mode = not extended_mode
            continue
            
        if user_input == "0": await kernel_menu()
        elif user_input == "1": await swarm_menu()
        elif user_input == "2": await models_menu()
        elif user_input == "3": await research_menu()
        elif user_input == "4": await opts_menu()
        elif user_input == "5": await intelligence_labs_menu()
        elif user_input == "6": await handle_messaging_apps()
        elif user_input == "7": await system_menu()
        elif user_input == "9": await blockchain_menu()
        elif user_input == "10": await infrastructure_menu()
        elif user_input == "11": await task_registry_menu()
        elif user_input == "13": await marketing_intelligence_menu()
        elif user_input == "15": await embodied_rl_menu()
        elif user_input == "16": await handle_overdrive_menu()
        elif user_input.lower() == "p": await handle_personalize()
        elif user_input.lower() == "exit": break
        else:
            # Handle executive reasoning or other commands
            from interface.comm_menu import handle_executive_prompt
            try:
                await handle_executive_prompt(user_input)
            except ImportError:
                console.print(f"[yellow]Command '{user_input}' not recognized.[/yellow]")

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted. Exiting...[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Critical Error: {e}[/bold red]")
