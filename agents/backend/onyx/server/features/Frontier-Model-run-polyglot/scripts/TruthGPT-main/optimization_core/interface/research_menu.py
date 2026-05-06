"""
SOTA Research & Deep Discovery Hub
"""
import time
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

from interface.core import (
    console, USER_PREFS, clear_screen, get_header, wait_for_user
)

async def research_menu():
    from modules.base.core_system.core.papers.paper_registry import PaperRegistry
    registry = PaperRegistry()
    while True:
        clear_screen()
        console.print(get_header())
        papers = registry.list_papers()[:10]
        console.print(f" [bold magenta]SOTA Trend Radar:[/bold magenta] [dim]{len(papers)} papers indexed[/dim]")
        for i, p in enumerate(papers, 1):
            console.print(f" {i:2} | [magenta]{p.paper_id:25}[/magenta] | [green]{p.category}[/green]")
        console.print(" [bold white]D[/bold white] | Autonomous Discovery (ArXiv)")
        console.print(" [bold white]T[/bold white] | Tavily Neural Search (SOTA)")
        console.print(" [bold white]M[/bold white] | Mathematical Discovery (Erdos Solver)")
        console.print(" [bold white]0[/bold white] | Return")
        choice = Prompt.ask("Selection").upper()
        if choice == "0": break
        elif choice == "M":
            from modules.math.erdos_solver import ErdosSolver
            solver = ErdosSolver()
            solver.forensic_report()
        elif choice == "T":
            query = Prompt.ask("Research Query")
            console.print(f"[cyan]➤ Querying Tavily for {query}...[/cyan]")
            time.sleep(1)
        wait_for_user(force=True)

async def intelligence_labs_menu():
    labs = [("Data Analysis", "data_expert"), ("Reasoning Lab", "reasoning_agent")]
    while True:
        clear_screen()
        console.print(get_header())
        lab_table = Table(title="🧠 Intelligence Labs")
        for i, (name, _) in enumerate(labs, 1): lab_table.add_row(str(i), name)
        console.print(lab_table)
        choice = Prompt.ask("Selection", choices=["0", "1", "2"])
        if choice == "0": break
        # ... logic ...
        wait_for_user(force=True)
