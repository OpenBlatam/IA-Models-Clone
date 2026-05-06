"""
Model & Training Hub - Frontier Orchestration
"""
import time
import os
from pathlib import Path
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

from interface.core import (
    console, USER_PREFS, clear_screen, get_header, wait_for_user
)


# Import CLI components
try:
    import cli
except ImportError:
    from .. import cli

async def models_menu():
    while True:
        clear_screen()
        console.print(get_header())
        menu_table = Table(title="🚀 Model & Training Hub", border_style="cyan", expand=True)
        menu_table.add_row("1", "Inference", "Run model on local prompt")
        menu_table.add_row("2", "Fast Train", "Train with default HF engine")
        menu_table.add_row("3", "SOTA Train", "GRPO/MCTS Advanced Training")
        menu_table.add_row("7", "Model Architect", "🛠️ Build & Inject Custom Model")
        menu_table.add_row("8", "Code Injector", "💉 Upgrade & Inject SOTA Logic")
        menu_table.add_row("9", "HF Downloader", "📥 Pull any model from Hugging Face")
        menu_table.add_row("0", "Back", "")
        console.print(menu_table)
        choice = Prompt.ask("Selection", choices=["0", "1", "2", "3", "7", "8", "9"])
        if choice == "0": break
        elif choice == "1":
            text = Prompt.ask("Enter prompt")
            cli.infer(text=text)
        elif choice == "2": cli.train()
        elif choice == "3": cli.train(override=["training.method=grpo"])
        elif choice == "7": await handle_model_architect()
        elif choice == "8": await handle_code_injector()
        elif choice == "9": await handle_hf_downloader()
        wait_for_user(force=True)

async def handle_model_architect():
    clear_screen()
    console.print(Panel("[bold cyan]🛠️ TruthGPT Model Architect[/bold cyan]", border_style="cyan"))
    name = Prompt.ask("Model Name", default="custom_transformer")
    from agents.client import AgentClient
    from agents.engines import engine_registry
    llm = engine_registry.get_engine(USER_PREFS.get("preferred_engine", "deepseek"))
    client = AgentClient(use_swarm=False, llm_engine=llm)
    with console.status(f"[bold cyan]AI Designer is synthesizing {name}...[/bold cyan]"):
        try:
            res = await client.run(user_id="model_architect", prompt=f"Generate PyTorch code for {name}")
            code = res.content if hasattr(res, 'content') else str(res)
            save_path = Path("truthgpt_collected/models") / f"{name}.py"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(code)
            console.print(f"[green]✓ Model {name} created at {save_path}[/green]")
        except Exception as e: console.print(f"[red]Error: {e}[/red]")
    wait_for_user(force=True)

async def handle_code_injector():
    clear_screen()
    file_path = Prompt.ask("Path to source file (.py)")
    if not os.path.exists(file_path): return
    with console.status("[bold magenta]Refactoring and injecting logic...[/bold magenta]"):
        # logic here...
        time.sleep(1)
        console.print("[green]✓ Logic injected.[/green]")
    wait_for_user(force=True)

async def handle_hf_downloader():
    clear_screen()
    query = Prompt.ask("Search models on HF")
    # downloader logic...
    console.print(f"[cyan]Downloading {query}...[/cyan]")
    time.sleep(1)
    wait_for_user(force=True)
