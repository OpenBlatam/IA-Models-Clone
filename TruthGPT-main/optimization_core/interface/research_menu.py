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
from interface.cc_style import cc_menu, cc_step
from rich import box

@cc_menu("SOTA Research & Deep Discovery")
async def research_menu():
    from modules.base.core_system.core.papers.paper_registry import get_paper_registry
    registry = get_paper_registry(preload_popular=False)
    while True:
        clear_screen()
        console.print(get_header())
        papers = registry.list_papers()[:10]
        
        table = Table(title="[bold cyan]SOTA Trend Radar: Indexed Papers[/bold cyan]", box=box.ROUNDED, expand=True)
        table.add_column("Idx", justify="center", style="dim", width=4)
        table.add_column("Paper ID", style="magenta", width=25)
        table.add_column("Title", style="white")
        table.add_column("Category", justify="center", style="green")
        table.add_column("Boost", justify="right", style="yellow")
        
        for i, p in enumerate(papers, 1):
            boost_str = f"⚡ {getattr(p, 'speedup', '1.0')}x 🎯 +{getattr(p, 'accuracy_improvement', '0.0')}%"
            title_str = getattr(p, 'title', 'Unknown Title')
            if len(title_str) > 50:
                title_str = title_str[:47] + "..."
            table.add_row(str(i), p.paper_id, title_str, p.category, boost_str)
            
        console.print(table)
        
        options_panel = Panel(
            "[bold white]D[/bold white] | 🤖 Autonomous Discovery (ArXiv)\n"
            "[bold white]T[/bold white] | 🌐 Tavily Neural Search (SOTA)\n"
            "[bold white]M[/bold white] | ♾️  Mathematical Discovery (Erdos Solver)\n"
            "[bold white]B[/bold white] | 🏆 Auto-Integrate Best of the Month (TruthGPT)\n"
            "[bold white]0[/bold white] | ⬅️  Return",
            title="[bold cyan]Operations & Actions[/bold cyan]",
            border_style="cyan",
            padding=(0, 2)
        )
        console.print(options_panel)
        choice = Prompt.ask("Selection").upper()
        if choice == "0": break
        elif choice == "B":
            from agents.system_intelligence.system_tools import PaperSynthesisTool
            # Find the best paper in the registry
            best_paper = None
            best_score = -1.0
            for p in registry.list_papers():
                try:
                    speedup = float(getattr(p, 'speedup', 1.0))
                    acc = float(getattr(p, 'accuracy_improvement', 0.0))
                    score = speedup * 2 + acc # weight speedup and accuracy
                    if score > best_score:
                        best_score = score
                        best_paper = p
                except:
                    pass
            
            if not best_paper:
                console.print("[red]No papers available to integrate.[/red]")
                wait_for_user(force=True)
                continue
                
            console.print(Panel(f"[bold magenta]{best_paper.title}[/bold magenta]\n[green]Category:[/green] {best_paper.category}\n[yellow]Boost Score:[/yellow] {best_score:.2f} (Speedup: {getattr(best_paper, 'speedup', 1.0)}x, Acc: +{getattr(best_paper, 'accuracy_improvement', 0.0)}%)", title="🏆 Best Paper of the Month", border_style="yellow"))
            
            import subprocess
            import sys
            from pathlib import Path
            with console.status(f"[bold magenta]Auto-Integrating Best Paper {best_paper.paper_id} to TruthGPT...[/bold magenta]"):
                p_id_clean = best_paper.paper_id.replace(".", "_").replace("-", "_")
                script_path = Path(f"optimization_core/truthgpt_collected/integration_code/papers/research/paper_{p_id_clean}.py")
                if not script_path.exists():
                    synthesis = PaperSynthesisTool()
                    await synthesis.run(f"{best_paper.paper_id}:::{best_paper.title}:::{best_paper.category}:::N/A")
                
                try:
                    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True, timeout=45)
                    success, output = result.returncode == 0, result.stdout + result.stderr
                except Exception as e: success, output = False, str(e)
            
            if success:
                console.print(Panel(f"[bold green]✓ Best Paper Applied Successfully into TruthGPT[/bold green]\n\n{output[-500:]}", border_style="green"))
            else:
                console.print(Panel(f"[bold red]✗ Application Failed[/bold red]\n\n{output}", border_style="red"))
            wait_for_user(force=True)
            
        elif choice.isdigit() and 1 <= int(choice) <= len(papers):
            selected = papers[int(choice)-1]
            clear_screen()
            console.print(Panel(f"[bold magenta]Paper Selection:[/bold magenta] {selected.paper_id}", border_style="magenta"))
            console.print(f"[bold]Title:[/bold] {selected.title}")
            console.print(f"[bold]Category:[/bold] {selected.category}")
            console.print(f"[bold]ArXiv ID:[/bold] {getattr(selected, 'arxiv_id', 'N/A')}")
            
            action = Prompt.ask("\n[1] View Info [2] Apply/Execute [0] Back", choices=["0", "1", "2"])
            if action == "1":
                from optimization_core.modules.base.core_system.core.papers.paper_registry import PaperRegistry
                registry = PaperRegistry()
                paper = next((p for p in registry.list_papers() if p.paper_id == selected.paper_id), None)
                if paper:
                    link = f"https://arxiv.org/abs/{paper.arxiv_id}" if getattr(paper, 'arxiv_id', None) else "N/A"
                    console.print(Panel(f"[bold]Paper ID:[/bold] {paper.paper_id}\n[bold]Category:[/bold] {paper.category}\n[bold]SOTA Link:[/bold] {link}\n[bold]Techniques:[/bold] {', '.join(paper.key_techniques) if getattr(paper, 'key_techniques', None) else 'N/A'}\n[bold]Speedup:[/bold] {getattr(paper, 'speedup', '1.0')}x\n[bold]Accuracy:[/bold] +{getattr(paper, 'accuracy_improvement', '0.0')}%", title=f"📄 Paper: {paper.title}", border_style="magenta"))
                wait_for_user(force=True)
            elif action == "2":
                import subprocess
                import sys
                from pathlib import Path
                with console.status(f"[bold magenta]Applying Paper {selected.paper_id}...[/bold magenta]"):
                    p_id_clean = selected.paper_id.replace(".", "_").replace("-", "_")
                    script_path = Path(f"optimization_core/truthgpt_collected/integration_code/papers/research/paper_{p_id_clean}.py")
                    if not script_path.exists():
                        from agents.system_intelligence.system_tools import PaperSynthesisTool
                        synthesis = PaperSynthesisTool()
                        await synthesis.run(f"{selected.paper_id}:::{selected.title}:::{selected.category}:::N/A")
                    
                    try:
                        result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True, timeout=30)
                        success, output = result.returncode == 0, result.stdout + result.stderr
                    except Exception as e: success, output = False, str(e)
                
                if success:
                    console.print(Panel(f"[bold green]✓ Paper Applied Successfully[/bold green]\n\n{output[-500:]}", border_style="green"))
                else:
                    console.print(Panel(f"[bold red]✗ Application Failed[/bold red]\n\n{output}", border_style="red"))
                wait_for_user(force=True)

        elif choice == "D":
            query = Prompt.ask("Search ArXiv (e.g., 'Transformer Optimization')")
            if not query or query.strip() == "0":
                continue
            
            while query and query.strip() != "0":
                # --- Perform ArXiv Search ---
                import httpx
                import xml.etree.ElementTree as ET
                from rich.table import Table
                
                with console.status(f"[bold magenta]Searching ArXiv for '{query}'...[/bold magenta]"):
                    search_query = f"all:{query.replace(' ', '+')}"
                    url = f"https://export.arxiv.org/api/query?search_query={search_query}&max_results=10"
                    found_papers = []
                    try:
                        import httpx
                        import time
                        for attempt in range(3):
                            response = httpx.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
                            if response.status_code == 200:
                                break
                            if response.status_code == 429:
                                time.sleep(3)
                                continue
                            break
                        
                        if response.status_code != 200:
                            raise Exception(f"HTTP {response.status_code}: {response.text[:100]}")
                            
                        root = ET.fromstring(response.text)
                        ns = {'atom': 'http://www.w3.org/2005/Atom'}
                        for entry in root.findall('atom:entry', ns):
                            t_elem = entry.find('atom:title', ns)
                            title = t_elem.text.strip().replace('\n', ' ') if t_elem is not None else "Unknown Title"
                            i_elem = entry.find('atom:id', ns)
                            arxiv_id = i_elem.text.split('/')[-1] if i_elem is not None else "Unknown"
                            c_elem = entry.find('atom:category', ns)
                            category = c_elem.attrib.get('term', 'Unknown') if c_elem is not None else "Unknown"
                            found_papers.append({"id": arxiv_id, "title": title, "category": category})
                    except Exception as e:
                        console.print(f"[red]Error searching ArXiv: {e}[/red]")
                        wait_for_user(force=True)
                        break

                if not found_papers:
                    console.print("[yellow]No papers found for that query.[/yellow]")
                    query = Prompt.ask("Search ArXiv (e.g., 'Transformer Optimization') or '0' to return")
                    continue

                clear_screen()
                console.print(Panel(f"[bold magenta]ArXiv Search Results for:[/bold magenta] {query}", border_style="magenta"))
                
                results_table = Table(box=None)
                results_table.add_column("Idx", style="dim", width=4)
                results_table.add_column("ID", style="cyan", width=15)
                results_table.add_column("Title", style="white")
                
                for i, p in enumerate(found_papers, 1):
                    results_table.add_row(str(i), p["id"], p["title"])
                
                console.print(results_table)
                
                sub_choice = Prompt.ask("\nEnter # to adopt, a new query, or '0' to return")
                
                if not sub_choice or sub_choice.strip() == "0":
                    break
                
                target_paper = None
                paper_id = None
                if sub_choice.isdigit() and 1 <= int(sub_choice) <= len(found_papers):
                    target_paper = found_papers[int(sub_choice)-1]
                    paper_id = target_paper["id"]
                    paper_title = target_paper["title"]
                    query = None
                else:
                    # Let's check if the sub_choice is a valid ArXiv ID pattern.
                    import re
                    is_arxiv_id = re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", sub_choice.strip()) or re.match(r"^[a-zA-Z\-]+(\.[a-zA-Z\-]+)?/\d{7}(v\d+)?$", sub_choice.strip())
                    if is_arxiv_id:
                        paper_id = sub_choice.strip()
                        paper_title = f"Manual Discovery: {paper_id}"
                        query = None
                    else:
                        query = sub_choice
                        continue

                if paper_id:
                    from agents.system_intelligence.system_tools import PaperSynthesisTool, SOTAPaperScraperTool
                    with console.status(f"[bold cyan]Scraping Paper {paper_id}...[/bold cyan]"):
                        try:
                            scraper = SOTAPaperScraperTool()
                            scrape_res = await scraper.run(paper_id)
                            console.print(f"[dim]{scrape_res}[/dim]")
                        except Exception as e:
                            console.print(f"[red]Error scraping paper {paper_id}: {e}[/red]")
                            wait_for_user(force=True)
                            break
                    
                    with console.status(f"[bold green]Synthesizing Implementation for {paper_id}...[/bold green]"):
                        try:
                            synthesis = PaperSynthesisTool()
                            synth_res = await synthesis.run(f"{paper_id}:::{paper_title}:::Deep Learning:::Synthesized from ArXiv Discovery")
                            console.print(Panel(synth_res, title="Integration Result", border_style="green"))
                        except Exception as e:
                            console.print(f"[red]Error synthesizing paper {paper_id}: {e}[/red]")
                    wait_for_user(force=True)
                    break

        elif choice == "M":
            from modules.math.erdos_solver import ErdosSolver
            solver = ErdosSolver()
            solver.forensic_report()
            wait_for_user(force=True)
        elif choice == "T":
            query = Prompt.ask("Research Query")
            if query:
                from optimization_core.utils.internet_search import search_internet
                from rich.table import Table
                from rich.panel import Panel
                
                with console.status(f"[bold cyan]➤ Querying Internet for '{query}'...[/bold cyan]"):
                    try:
                        results = await search_internet(query, max_results=5)
                    except Exception as e:
                        results = []
                        console.print(f"[red]Error performing web search: {e}[/red]")
                
                if results:
                    clear_screen()
                    console.print(Panel(f"[bold magenta]Web Search Results for:[/bold magenta] {query}", border_style="magenta"))
                    
                    table = Table(box=None)
                    table.add_column("Idx", style="dim", width=4)
                    table.add_column("Title", style="white bold")
                    table.add_column("Link", style="cyan")
                    
                    for i, r in enumerate(results, 1):
                        table.add_row(str(i), r["title"], r["link"])
                    
                    console.print(table)
                    console.print("\n[bold magenta]Details:[/bold magenta]")
                    for i, r in enumerate(results, 1):
                        console.print(f"\n[bold cyan][{i}] {r['title']}[/bold cyan]")
                        console.print(f"[dim]{r['link']}[/dim]")
                        console.print(f"{r['snippet']}")
                else:
                    console.print("[yellow]No results found on the internet.[/yellow]")
            wait_for_user(force=True)

@cc_menu("Intelligence Labs")
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
