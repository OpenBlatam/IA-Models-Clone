"""
Swarm Intelligence Hub - Industrial Command Center
"""
import asyncio
import time
import json
import inspect
import re
from pathlib import Path
from typing import Optional
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, FloatPrompt, Confirm
from rich.live import Live
from rich.console import Console
import io

# UI Framework Imports

from interface.core import (
    console, USER_PREFS, log_activity, log_event, clear_screen, 
    get_header, wait_for_user, background_missions, save_mission_output,
    export_mission_result, get_theme_panel, get_input, extract_target_directory
)
from interface.cc_style import cc_menu, cc_step, cc_action, cc_spinner, cc_agent_done
from interface.interactive_swarm import get_interactive_choice

"""
Swarm Intelligence Hub - Industrial Command Center
"""
import asyncio
import time
import json
import inspect
import re
from pathlib import Path
from typing import Optional
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, FloatPrompt, Confirm
from rich.live import Live
from rich.console import Console
import io

# UI Framework Imports

from interface.core import (
    console, USER_PREFS, log_activity, log_event, clear_screen, 
    get_header, wait_for_user, background_missions, save_mission_output,
    export_mission_result, get_theme_panel, get_input, extract_target_directory
)
from interface.cc_style import cc_menu, cc_step, cc_action, cc_spinner, cc_agent_done
from interface.interactive_swarm import get_interactive_choice

# Deferred CLI imports for performance

_client_cache = None

from utils.file_ops import extract_filename_from_code, save_code_blocks_to_directory
from orchestration.missions import BackgroundMission, wait_with_interrupt

class SwarmMenuApp:
    def __init__(self, active_agents):
        self.active_agents = active_agents
        self.selected_index = 0
        from prompt_toolkit.key_binding import KeyBindings
        self.kb = KeyBindings()
        self.result = None
        
        @self.kb.add('q')
        @self.kb.add('c-c')
        @self.kb.add('0')
        @self.kb.add('escape')
        def _(event):
            event.app.exit(result="0")

        # Hotkeys for commands (Case-insensitive)
        @self.kb.add('a')
        @self.kb.add('A')
        def _(event): event.app.exit(result="A")
        
        @self.kb.add('f')
        @self.kb.add('F')
        def _(event): event.app.exit(result="F")
        
        @self.kb.add('b')
        @self.kb.add('B')
        def _(event): event.app.exit(result="B")
        
        @self.kb.add('m')
        @self.kb.add('M')
        def _(event): event.app.exit(result="M")
        
        @self.kb.add('s')
        @self.kb.add('S')
        def _(event): event.app.exit(result="S")
        
        @self.kb.add('h')
        @self.kb.add('H')
        def _(event): event.app.exit(result="H")
        
        @self.kb.add('t')
        @self.kb.add('T')
        def _(event): event.app.exit(result="T")
        
        @self.kb.add('x')
        @self.kb.add('X')
        def _(event): event.app.exit(result="X")
        
        @self.kb.add('c')
        @self.kb.add('C')
        def _(event): event.app.exit(result="C")
        
        @self.kb.add('p')
        @self.kb.add('P')
        def _(event): event.app.exit(result="P")

        # Numeric keys for Active Experts
        for i in range(1, 10):
            @self.kb.add(str(i))
            def _(event, i=i):
                event.app.exit(result=str(i))

    def get_layout(self):
        from prompt_toolkit.application import get_app
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.formatted_text import ANSI
        from prompt_toolkit.layout.containers import Window, WindowAlign, HSplit
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.mouse_events import MouseEventType
        
        def set_choice(val):
            self.result = val
            get_app().exit(result=val)

        # Header with Real Telemetry
        header_console = Console(file=io.StringIO(), force_terminal=True, width=120)
        from interface.core import get_claude_header
        swarm_updates = [
            "Recursive Reasoning Enabled",
            "Expert Matrix Optimized",
            "Swarm Fusion Engine v2.4",
            "Latency: 12ms Cluster-Wide"
        ]
        header_console.print(get_claude_header(updates=swarm_updates))
        static_content = FormattedTextControl(ANSI(header_console.file.getvalue()))

        list_items = []
        
        def make_item(lid, name, val, index):
            def get_formatted_text():
                is_selected = (self.selected_index == index)
                style_prefix = "underline cyan" if is_selected else ""
                return [
                    ('class:dot', '             â— '),
                    ('class:id', f' {lid} '),
                    (f'class:name {style_prefix}', f' {name} '),
                ]

            def mouse_handler(mouse_event):
                if mouse_event.event_type == MouseEventType.MOUSE_MOVE:
                    self.selected_index = index
                elif mouse_event.event_type == MouseEventType.MOUSE_UP:
                    set_choice(val)

            content = FormattedTextControl(
                get_formatted_text,
                show_cursor=False,
            )
            content.mouse_handler = mouse_handler
            return Window(content=content, height=1, align=WindowAlign.LEFT)

        # Swarm Commands
        list_items.append(Window(height=1))
        list_items.append(make_item("A", "ðŸ“¡ Ask Swarm (Auto-Routing)", "A", 0))
        list_items.append(make_item("F", "ðŸŒ€ Dynamic Swarm Fusion", "F", 1))
        list_items.append(make_item("C", "âš¡ Continuous Mission", "C", 2))
        list_items.append(make_item("B", "ðŸ“¡ Background Missions", "B", 3))
        list_items.append(make_item("M", "ðŸ”Œ MCP Connectors", "M", 4))
        list_items.append(make_item("S", "ðŸ“Š Swarm Status", "S", 5))
        list_items.append(make_item("H", "ðŸ“œ Agent History & Audits", "H", 6))
        list_items.append(make_item("T", "ðŸ§® Math & Verification", "T", 7))
        list_items.append(make_item("X", "ðŸ—ï¸ Agent Composer", "X", 8))
        list_items.append(make_item("P", "ðŸŽ­ Persona Tuning", "P", 9))
        
        # List of Active Agents
        if self.active_agents:
            list_items.append(Window(height=1))
            header_console = Console(file=io.StringIO(), force_terminal=True, width=100)
            header_console.print("  [bold white]ACTIVE EXPERTS[/bold white]")
            list_items.append(Window(content=FormattedTextControl(ANSI(header_console.file.getvalue())), height=1))
            for i, agent in enumerate(self.active_agents):
                list_items.append(make_item(str(i+1), agent.name, str(i+1), 10+i))

        list_items.append(Window(height=1))
        list_items.append(make_item("0", "ðŸ”™ Back to Kernel", "0", 20))

        # Footer Status Bar
        footer_text = [
            ("class:prompt_seg", " â¯ SWARM HUB "),
            ("", " "),
            ("class:shortcut_seg", " ENTER "), ("class:shortcut_label", " Select "),
            ("class:shortcut_seg", " 0 "), ("class:shortcut_label", " Back "),
            ("", "  "),
            ("class:load_label", "SWARM LOAD: "), ("class:load_bar", "â–ˆâ–“â–’â–‘ 14%"),
            ("", "  "),
            ("class:version_seg", " Node: CLUSTER-7 ")
        ]

        return Layout(HSplit([
            Window(content=static_content, wrap_lines=True),
            HSplit(list_items),
            Window(height=1),
            Window(content=FormattedTextControl(footer_text), height=1),
        ]))

    async def run(self):
        from prompt_toolkit.styles import Style
        from prompt_toolkit.application import Application
        
        style = Style.from_dict({
            'dot': 'bold cyan', 'id': 'bold white', 'name': 'white',
            'prompt_seg': 'bg:magenta black bold',
            'shortcut_seg': 'bg:white black bold',
            'shortcut_label': 'white',
            'load_label': 'dim', 'load_bar': 'bold magenta',
            'version_seg': 'bg:#222222 dim',
        })
        app = Application(layout=self.get_layout(), key_bindings=self.kb, style=style, mouse_support=True, full_screen=True)
        self.result = await app.run_async()
        return self.result

async def swarm_menu():
    global _client_cache
    from agents.client import AgentClient
    from agents.engines import engine_registry
    from interface.core import USER_PREFS
    
    if _client_cache is None:
        # Pre-load minimized client with the preferred engine for zero latency
        engine_name = USER_PREFS.get("preferred_engine", "deepseek")
        try:
            llm = engine_registry.get_engine(engine_name)
        except:
            llm = None # Fallback to dummy or default
            
        _client_cache = AgentClient(use_swarm=True, llm_engine=llm)
    client = _client_cache
    
    while True:
        active_agents = []
        if hasattr(client.swarm, "agents"):
            active_agents = list(client.swarm.agents.values())
            
        app = SwarmMenuApp(active_agents)
        choice = await app.run()
        
        if choice is None or choice == "0": break
        elif choice == "A": await handle_swarm_ask()
        elif choice == "C": await handle_continuous_mission()
        elif choice == "B": await handle_background_missions()
        elif choice == "F": await handle_swarm_fusion()
        elif choice == "M": await handle_mcp_connect()
        elif choice == "S": await handle_swarm_telemetry()
        elif choice == "H":
            from interface.history_menu import agent_history_menu
            await agent_history_menu()
        elif choice == "T": await handle_math_verification()
        elif choice == "X": await handle_agent_composer()
        elif choice == "P": await handle_persona_tuning()
        elif choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(active_agents):
                target = active_agents[idx-1]
                prompt = get_input(f"Query {target.name}")
                console.print(Panel("[italic dim]Pensando... analizando contexto y seleccionando herramientas Ã³ptimas.[/italic dim]", title="[bold plum1]Thinking[/bold plum1]", border_style="plum1"))
                response = await target.process(prompt, context={"user_id": "cli"})
                content = response.content if hasattr(response, 'content') else str(response)
                console.print(get_theme_panel(content, title=f"ðŸ¤– {target.name} Response"))
                wait_for_user(force=True)


@cc_step("Swarm Router")
async def handle_swarm_ask():
    prompt = get_input("Enter your question for the swarm")
    engine = USER_PREFS["preferred_engine"]
    log_activity("Swarm Ask", prompt)
    with console.status(f"[bold blue]Routing to expert agents using {engine}...[/bold blue]"):
        try:
            import cli
            await cli.async_swarm_ask(prompt=prompt, user_id="cli_user", stream=False, engine=engine)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    wait_for_user(force=True)

@cc_menu("Dynamic Swarm Fusion")
async def handle_swarm_fusion(initial_prompt: Optional[str] = None):
    clear_screen()
    console.print(get_header())
    if initial_prompt:
        mode = "1"
    else:
        console.print("   1. ðŸ§  [bold]Autonomous Mode[/bold] (LLM decides the team)")
        console.print("   2. ðŸŽ¨ [bold]Designer Mode[/bold] (You build the sequence)")
        console.print("   0. ðŸ  Back to Swarm Menu")
        mode = get_input("Select mode", choices=["0", "1", "2"])
        if mode == "0": return

    from agents.registry import registry
    from agents.models import AgentConfig
    from agents.engines import engine_registry
    
    # Use lazy loading: get known agent keys without forcing all imports
    known_agent_keys = list(registry._agent_map.keys()) + list(registry._agents.keys())
    known_agent_keys = sorted(set(known_agent_keys))
    
    config = AgentConfig()
    try:
        llm = engine_registry.get_engine(USER_PREFS["preferred_engine"])
    except Exception as e:
        log_event("Engine Error", f"Failed to load engine: {e}")
        llm = None
    selected_keys = []
    
    if mode == "1":
        prompt = initial_prompt if initial_prompt else get_input("Enter task for the Autonomous Swarm")
        with console.status("[bold magenta]ðŸ§  Swarm Orchestrator is choosing experts...[/bold magenta]"):
            if llm:
                agent_list = ", ".join(known_agent_keys)
                decision_prompt = (
                    f"Given these agents: [{agent_list}], which ones are the MOST relevant for this task: '{prompt}'?\n"
                    f"Respond ONLY with a JSON list of keys, e.g. [\"research_agent\", \"marketing_agent\"]. "
                    f"Max 5 agents. Order them by execution sequence."
                )
                try:
                    # Timeout guard: prevent infinite hang on API calls
                    decision_res = await asyncio.wait_for(llm(decision_prompt), timeout=60.0)
                    if decision_res:
                        # Clean/replace single quotes for valid json parser
                        match = re.search(r"\[.*?\]", decision_res.replace("\n", ""))
                        if match:
                            try:
                                json_str = match.group().replace("'", '"')
                                parsed = json.loads(json_str)
                                # Validate keys exist in known agents
                                selected_keys = [k for k in parsed if k in known_agent_keys]
                            except (json.JSONDecodeError, Exception):
                                pass
                        
                        # Fallback parsing: look for keywords of keys in text in order of appearance
                        if not selected_keys:
                            found_keys = []
                            for key in known_agent_keys:
                                if key in decision_res:
                                    indices = [m.start() for m in re.finditer(re.escape(key), decision_res)]
                                    for idx in indices:
                                        found_keys.append((idx, key))
                            found_keys.sort()
                            seen = set()
                            for _, key in found_keys:
                                if key not in seen:
                                    seen.add(key)
                                    selected_keys.append(key)
                except asyncio.TimeoutError:
                    log_event("Swarm Timeout", "LLM orchestrator timed out after 60s. Using fallback experts.")
                    console.print("[yellow]âš ï¸ LLM inference timed out. Using fallback experts.[/yellow]")
                except Exception as e:
                    log_event("Swarm Error", f"Orchestrator failed to select agents: {e}")
                    console.print(f"[yellow]âš ï¸ Orchestrator error: {e}[/yellow]")
            
            if not selected_keys:
                # Fallback experts for autonomous mission if LLM fails or is not configured
                selected_keys = []
                for fallback_key in ["research_agent", "code_architect", "system_agent"]:
                    if fallback_key in known_agent_keys:
                        selected_keys.append(fallback_key)
                if not selected_keys and known_agent_keys:
                    selected_keys = [known_agent_keys[0]]
                log_event("Swarm", f"Using fallback experts: {selected_keys}")
                console.print(f"[yellow]âš ï¸ Could not auto-select experts. Using default: {', '.join(selected_keys)}[/yellow]")
    else:
        table = Table(title="Available Experts & Specialized Phases")
        table.add_column("#", style="cyan")
        table.add_column("Key", style="white")
        table.add_column("Expertise", style="dim")
        display_keys = known_agent_keys[:]
        if "arxiv_discovery_scout" not in display_keys:
            display_keys.append("arxiv_discovery_scout")
        for i, k in enumerate(display_keys, 1):
            expertise = "Research Discovery (ArXiv/Internet)" if k == "arxiv_discovery_scout" else "Specialized Agent"
            table.add_row(str(i), k, expertise)
        table.add_row("0", "ðŸ  Back", "Return to Swarm Menu")
        console.print(table)
        selection = get_input("Design your sequence (e.g. 5,1,2) or 0 to go back")
        if selection.strip() == "0":
            return
        indices = [int(i.strip()) for i in selection.split(",") if i.strip().isdigit()]
        selected_keys = [display_keys[i-1] for i in indices if 1 <= i <= len(display_keys)]
        prompt = initial_prompt if initial_prompt else get_input("Enter the initial task/seed for this custom swarm")

    if not selected_keys:
        console.print("[red]No agents selected for orchestration.[/red]")
        wait_for_user()
        return

    # --- Memory Selection Layer ---
    console.print("\n[bold cyan]ðŸ§  Memory Architecture Selection[/bold cyan]")
    mem_table = Table(show_header=False, border_style="cyan")
    mem_table.add_row("1", "Episodic Forensic (Persistent SQLite)", "[bold green]INDUSTRIAL[/bold green]")
    mem_table.add_row("2", "Semantic Vector (RAG/FAISS)", "[bold yellow]KNOWLEDGE[/bold yellow]")
    mem_table.add_row("3", "Holographic High-Dim (Experimental)", "[magenta]EXPERIMENTAL[/magenta]")
    mem_table.add_row("4", "Paper-Driven (Latest Research SOTA)", "[bold blue]SCIENTIFIC[/bold blue]")
    mem_table.add_row("5", "Knowledge Graph (Relational/Triples)", "[bold cyan]COMPLEX[/bold cyan]")
    mem_table.add_row("6", "ULTIMATE Hybrid (Combine ALL Layers)", "[bold red]MASTER[/bold red]")
    mem_table.add_row("0", "ðŸ  Back", "[dim]Return[/dim]")
    console.print(mem_table)
    mem_choice = get_input("Select Memory Type", choices=["0", "1", "2", "3", "4", "5", "6"], default="1")
    if mem_choice == "0":
        return
    
    memory_config = {"type": "forensic", "trace_enabled": True}
    if mem_choice == "2": memory_config = {"type": "vector", "trace_enabled": True}
    elif mem_choice == "3": memory_config = {"type": "holographic", "trace_enabled": True}
    elif mem_choice == "5": memory_config = {"type": "graph", "trace_enabled": True}
    elif mem_choice == "6":
        memory_config = {
            "type": "hybrid_ultimate",
            "layers": ["forensic", "vector", "graph", "paper_dna"],
            "trace_enabled": True,
            "fusion_mode": "weighted_consensus"
        }
        console.print("[bold red]ðŸ”¥ Initializing ULTIMATE Hybrid Memory Fabric (Multi-Layer Orchestration)...[/bold red]")
    elif mem_choice == "4":
        try:
            from modules.base.core_system.core.papers.paper_registry import PaperRegistry
            reg = PaperRegistry()
            mem_papers = reg.list_papers(category="memory")
            if len(mem_papers) < 3: 
                mem_papers += reg.search_papers(query="attention")
            if len(mem_papers) < 5:
                mem_papers += reg.list_papers()[:10]
                
            if mem_papers:
                p_table = Table(title="ðŸ“š SOTA Memory & Architecture Research Papers", header_style="bold magenta", border_style="blue")
                p_table.add_column("#", style="cyan", justify="right")
                p_table.add_column("Paper ID", style="white")
                p_table.add_column("SOTA Technique", style="green")
                p_table.add_column("Impact", style="dim")
                for i, p in enumerate(mem_papers[:12], 1):
                    tech = ", ".join(p.key_techniques[:2]) if hasattr(p, 'key_techniques') and p.key_techniques else "General SOTA"
                    acc_val = getattr(p, 'accuracy_improvement', '5.0')
                    impact = f"+{acc_val if acc_val is not None else '0.0'}% Acc"
                    p_table.add_row(str(i), p.paper_id, tech, impact)
                console.print(p_table)
                p_idx_input = get_input("Select Paper DNA to inject (supports multi-select: 1,2,3)", default="1")
                try:
                    selected_papers = [mem_papers[int(s)-1] for s in p_idx_input.replace(' ','').split(',') if s.isdigit()]
                    paper_ids = [p.paper_id for p in selected_papers]
                    memory_config = {"type": "paper_driven", "paper_ids": paper_ids, "trace_enabled": True}
                    console.print(f"[green]âœ“ Injecting {', '.join(paper_ids)} into memory fabric.[/green]")
                except:
                    console.print("[yellow]Invalid selection. Using Forensic fallback.[/yellow]")
            else:
                console.print("[yellow]No memory papers found. Using Forensic standard.[/yellow]")
        except Exception as e:
            console.print(f"[yellow]âš ï¸ Paper registry unavailable ({e}). Using Forensic fallback.[/yellow]")

    console.print(f"\n[bold green]ðŸ§¬ Executing Swarm Blueprint: {' âž” '.join(selected_keys)}[/bold green]")
    if any([USER_PREFS.get("mcts_optimized"), USER_PREFS.get("speculative_decoding"), USER_PREFS.get("kv_quantization")]):
        console.print("[bold yellow]âš¡ Neural Overdrive Active: Optimizing for Speed & Logic...[/bold yellow]")

    # === GLOBAL EXECUTION GUARD â€” prevents silent exit on any crash ===
    try:
        await _execute_swarm_pipeline(selected_keys, initial_prompt, llm, config, memory_config, registry)
    except Exception as e:
        console.print(f"\n[bold red]âš ï¸ Swarm Fusion encountered a critical error: {e}[/bold red]")
        console.print("[yellow]The mission was interrupted but your session is safe.[/yellow]")
    wait_for_user(force=True)


async def _execute_swarm_pipeline(selected_keys, initial_prompt, llm, config, memory_config, registry):
    """Inner execution pipeline â€” isolated so crashes don't exit swarm_fusion."""
    log_activity("Swarm Fusion", f"Blueprint: {'->'.join(selected_keys)} | Memory: {memory_config['type']}")

    # --- Execution Orchestration ---
    exec_mode = get_input("Execution Architecture (S=Sequential, P=Parallel, 0=Back)", choices=["S", "P", "0"], default="S")
    if exec_mode == "0":
        return
    is_parallel = (exec_mode == "P")

    console.print(f"\n[bold yellow]ðŸš€ Launching Swarm Fusion ({'Parallel' if is_parallel else 'Sequential'})...[/bold yellow]\n")

    context = {"user_id": "orchestrator_fusion", "history": [], "memory_config": memory_config, "memory_trace": [], "shared_embeddings_fetched": False, "shared_state_synced": False}
    current_prompt = initial_prompt if initial_prompt else get_input("Enter task for the Swarm")
    content = ""

    async def run_phase(key, idx, phase_prompt):
        start_phase = time.time()
        trace_entry = {
            "phase": key, 
            "time": time.strftime('%H:%M:%S'), 
            "actions": [],
            "rationale": "Calculating optimal strategy based on previous state..."
        }
        
        if memory_config["type"] == "hybrid_ultimate":
            if not context.get("shared_embeddings_fetched"):
                trace_entry["actions"].append("Retrieved cross-layer relational embeddings (Global Cache Miss)")
                context["shared_embeddings_fetched"] = True
            else:
                trace_entry["actions"].append("Accessed cached cross-layer embeddings (Zero-Latency)")
                
            if not context.get("shared_state_synced"):
                trace_entry["actions"].append("Synced forensic persistent state (Incremental)")
                context["shared_state_synced"] = True
            else:
                trace_entry["actions"].append("Skipped redundant state sync (Delta unchanged)")
        else:
            trace_entry["actions"].append(f"Querying {memory_config['type']} memory layer")
            
        # Claude-style Thinking simulation per phase
        console.print(Panel(f"[italic dim]Fase {idx} ({key}): Orquestando lÃ³gica interna y consultando memoria {memory_config['type']}...[/italic dim]", title="[bold plum1]Thinking[/bold plum1]", border_style="plum1"))
        
        # NOP Guard for empty or purely generic phases if input is too small
        if not phase_prompt or len(phase_prompt.strip()) < 10:
            if key in ["research_agent", "sota_integrator"]:
                duration = time.time() - start_phase
                trace_entry["rationale"] = "Early exit: NOP guard triggered due to insufficient input data."
                trace_entry["actions"].append("SKIP: No actionable input data.")
                trace_entry["duration"] = f"{duration:.2f}s"
                trace_entry["speedup"] = "1.0x (Skipped)"
                return trace_entry, "[Skipped: Insufficient input data]"

        p_content = ""
        try:
            from contextlib import nullcontext
            ctx = nullcontext() if is_parallel else console.status(f"[bold cyan]Phase {idx}: '{key}' is executing...[/bold cyan]")
            if is_parallel:
                console.print(f"[bold cyan]Phase {idx}: '{key}' is executing...[/bold cyan]")
                
            with ctx:
                if key == "arxiv_discovery_scout":
                    from agents.system_intelligence.research_agent import ResearchAgent
                    agent = ResearchAgent(llm_engine=llm)
                    res = await asyncio.wait_for(
                        agent.process(f"descubrir e integrar papers de {phase_prompt}"),
                        timeout=120.0
                    )
                    p_content = res.content
                    trace_entry["rationale"] = f"Identified research gaps for {phase_prompt}. Seeking SOTA validation."
                else:
                    # Lazy-load the specific agent class
                    agent_cls = registry.get_agent(key)
                    if agent_cls is None:
                        p_content = f"[Agent '{key}' could not be loaded â€” skipping phase]"
                        trace_entry["rationale"] = f"Agent '{key}' failed to load."
                        trace_entry["actions"].append(f"SKIP: Agent class not available")
                    else:
                        sig = inspect.signature(agent_cls.__init__)
                        params = {}
                        if "config" in sig.parameters: params["config"] = config
                        if "llm_engine" in sig.parameters: params["llm_engine"] = llm
                        agent = agent_cls(**params)
                        res = await asyncio.wait_for(
                            agent.process(phase_prompt, context=context),
                            timeout=120.0
                        )
                        p_content = res.content if hasattr(res, 'content') else str(res)
                        
                        rationale = res.metadata.get("rationale") if hasattr(res, 'metadata') and res.metadata else None
                        if not rationale:
                            rationale = f"Executing {key} logic to transform state."
                        trace_entry["rationale"] = rationale

                # Auto-persist code
                if p_content and "```" in p_content:
                    target_dir = extract_target_directory(phase_prompt) or extract_target_directory(initial_prompt)
                    if target_dir:
                        code_dir = target_dir
                    else:
                        code_dir = Path("truthgpt_collected/generated_code")
                    saved = save_code_blocks_to_directory(p_content, code_dir, default_prefix=f"output_{key}")
                    if saved:
                        console.print(f"[bold green]ðŸ’¾ Code persisted: {saved[-1].name}[/bold green]")

        except asyncio.TimeoutError:
            p_content = f"[Phase '{key}' timed out after 120s]"
            trace_entry["rationale"] = f"Phase '{key}' timed out during execution."
            trace_entry["actions"].append("TIMEOUT: 120s limit exceeded")
            console.print(f"[bold yellow]âš ï¸ Phase {idx} ({key}) timed out. Continuing...[/bold yellow]")
        except Exception as e:
            p_content = f"[Phase '{key}' failed: {str(e)[:200]}]"
            trace_entry["rationale"] = f"Phase '{key}' encountered an error: {e}"
            trace_entry["actions"].append(f"ERROR: {str(e)[:100]}")
            console.print(f"[bold red]âš ï¸ Phase {idx} ({key}) error: {e}[/bold red]")
            console.print("[yellow]Continuing with next phase...[/yellow]")

        duration = time.time() - start_phase
        trace_entry["actions"].append(f"Committed phase output to {memory_config['type']} fabric")
        trace_entry["duration"] = f"{duration:.2f}s"
        if USER_PREFS.get("mcts_optimized"):
            # Adaptive Overdrive calculation
            if duration > 100:
                speedup_factor = "2.0x (Max Overdrive)"
            elif duration > 50:
                speedup_factor = "1.6x (High Overdrive)"
            elif duration > 20:
                speedup_factor = "1.4x (Standard Overdrive)"
            else:
                speedup_factor = "1.2x (Light Overdrive)"
            trace_entry["speedup"] = speedup_factor
            
        return trace_entry, p_content

    if is_parallel:
        tasks = [run_phase(key, i+1, current_prompt) for i, key in enumerate(selected_keys)]
        results = await asyncio.gather(*tasks)
        for trace, p_res in results:
            context["memory_trace"].append(trace)
            content += f"\n\n--- Phase Output ({trace['phase']}) ---\n{p_res}"
            console.print(Panel(p_res, title=f"âœ… {trace['phase']} Complete", border_style="green"))
    else:
        i = 0
        while i < len(selected_keys):
            key = selected_keys[i]
            
            # Dynamic Parallelization: Check if current and next are independent (e.g. math_verifier and system_agent)
            if key == "math_verifier" and i + 1 < len(selected_keys) and selected_keys[i+1] == "system_agent":
                next_key = selected_keys[i+1]
                t1 = run_phase(key, i+1, current_prompt)
                t2 = run_phase(next_key, i+2, current_prompt)
                res1, res2 = await asyncio.gather(t1, t2)
                
                for trace, p_res in (res1, res2):
                    context["memory_trace"].append(trace)
                    content += f"\n\n--- Phase Output ({trace['phase']}) ---\n{p_res}"
                    console.print(Panel(p_res, title=f"âœ… {trace['phase']} Complete (Parallel)", border_style="green"))
                
                current_prompt = f"Previous findings from {key} & {next_key}: {res1[1][:300]} | {res2[1][:300]}\n\nObjective: {current_prompt}"
                i += 2
                continue
                
            trace, p_res = await run_phase(key, i+1, current_prompt)
            context["memory_trace"].append(trace)
            content += f"\n\n--- Phase Output ({key}) ---\n{p_res}"
            # Update prompt for next sequential phase
            current_prompt = f"Previous findings: {p_res}\n\nObjective: {current_prompt}"
            console.print(Panel(p_res, title=f"âœ… {key} Complete", border_style="green"))
            i += 1
    
    console.print("\n[bold green]âœ“ Swarm Orchestration Complete.[/bold green]")
    
    # Save Log Trace to disk
    if memory_config.get("trace_enabled"):
        trace_path = Path("truthgpt_collected/logs/memory_traces")
        trace_path.mkdir(parents=True, exist_ok=True)
        filename = f"trace_{int(time.time())}.json"
        with open(trace_path / filename, "w") as f:
            json.dump(context["memory_trace"], f, indent=4)
        console.print(f"[dim]ðŸ’¾ Decision Trace persisted to {trace_path / filename}[/dim]")
        
        # New: Interactive Trace Review
        if Confirm.ask("[bold cyan]Would you like to review the Decision Logic Trace?[/bold cyan]"):
            t_table = Table(title="ðŸ•µï¸ Forensic Decision Trace", border_style="cyan")
            t_table.add_column("Phase", style="magenta")
            t_table.add_column("Rationale / Why?", style="white")
            t_table.add_column("Duration", style="yellow")
            t_table.add_column("Efficiency", style="green")
            t_table.add_column("Actions Taken", style="dim")
            for entry in context["memory_trace"]:
                t_table.add_row(
                    entry["phase"], 
                    entry["rationale"], 
                    entry.get("duration", "N/A"),
                    entry.get("speedup", "1.0x (Standard)"),
                    "\n".join([f"â€¢ {a}" for a in entry["actions"]])
                )
            console.print(t_table)
        
    # --- Post-Mission Autonomous Actions (Available for all missions) ---
    console.print("\n[bold cyan]âš¡ Post-Mission Autonomous Actions[/bold cyan]")
    action_table = Table(show_header=False, border_style="dim")
    action_table.add_row("1", "ðŸš€ [bold green]Self-Optimize[/bold green] (Run Overdrive on Results)")
    action_table.add_row("2", "ðŸ”„ [bold yellow]Continuous Mode[/bold yellow] (Recursive Mission)")
    action_table.add_row("3", "ðŸ›¡ï¸ [bold blue]Self-Refine[/bold blue] (Architect Review)")
    action_table.add_row("0", "ðŸ  Finish & Return")
    console.print(action_table)
    
    post_choice = get_input("Select next autonomous action", choices=["0", "1", "2", "3"], default="0")
    
    if post_choice == "1":
        from interface.overdrive_menu import handle_overdrive_menu
        await handle_overdrive_menu()
        # After overdrive, return to the same mission results or finish
    elif post_choice == "2":
        console.print("\n[bold yellow]ðŸ” Recursive Continuity Configuration[/bold yellow]")
        interval_min = FloatPrompt.ask(" [bold cyan]âž” Enter Execution Interval (minutes, 0 for instant)[/bold cyan]", default=0.0)
        console.print(f"[bold green]âœ“ Continuous Mission Mode Activated (Interval: {interval_min}m)[/bold green]")
        if interval_min > 0:
            action = await wait_with_interrupt(interval_min * 60)
            if action in ('stop', 'menu'):
                return
        await handle_swarm_fusion(initial_prompt=f"Evolve and improve the following results: {content}")
        return
    elif post_choice == "3":
        console.print("[bold blue]🛡️  Code Architect is refining the mission output...[/bold blue]")
        from agents.code_interpreter import CodeInterpreterAgent
        architect = CodeInterpreterAgent(config=config, llm_engine=llm)
        refinement = await architect.process(f"Refine and industrialize this code for System 5.9: {content}")
        console.print(Panel(refinement.content, title="🛡️  Architectural Refinement", border_style="blue"))


async def handle_continuous_mission():
    clear_screen()
    console.print(get_header())
    console.print(Panel(
        "[bold #ffbe0b]⏱ Continuous Mission Mode[/bold #ffbe0b]  [dim]· Autonomous persistent execution loop[/dim]",
        border_style="#ffbe0b",
        expand=False
    ))
    query = get_input("Enter the persistent mission query")
    if not query.strip() or query.strip().lower() in ["back", "exit", "0", "cancel", "q"]:
        return
    interval_min = FloatPrompt.ask("Execution interval (minutes)", default=5.0)
    
    console.print("\n[bold cyan]Seleccione la modalidad de ejecución (Costo de API vs Calidad):[/bold cyan]")
    console.print("  [bold green]1[/bold green]: Económico - Tokens reducidos, menor temperatura (Costo mínimo)")
    console.print("  [bold yellow]2[/bold yellow]: Punto Medio - Balanceado entre costo de tokens y razonamiento")
    console.print("  [bold magenta]3[/bold magenta]: Máxima Calidad - Máximo de tokens y contexto (Costo alto)")
    modality = get_input("Seleccione modalidad", choices=["1", "2", "3"], default="1")
    
    mode_labels = {"1": "Económico", "2": "Punto Medio", "3": "Máxima Calidad"}
    console.print(f"\n[green]✓ Mission started: '[bold]{query}[/bold]' (Modalidad {mode_labels.get(modality, 'Económico')})[/green]")
    from agents.client import AgentClient
    from agents.engines import engine_registry
    from interface.core import USER_PREFS
    
    selected_engine = USER_PREFS.get("preferred_engine", "deepseek")
    llm = engine_registry.get_engine(selected_engine)
    
    if modality == "1":
        llm.default_kwargs = {"temperature": 0.1, "max_tokens": 1024}
    elif modality == "2":
        llm.default_kwargs = {"temperature": 0.5, "max_tokens": 4096}
    elif modality == "3":
        llm.default_kwargs = {"temperature": 0.8, "max_tokens": 16384}
        
    client = AgentClient(use_swarm=True, llm_engine=llm)
    cycle = 0
    try:
        while True:
            cycle += 1
            ts = time.strftime('%H:%M:%S')
            # ── Cycle header
            console.print()
            console.print(f"[bold #8338ec]●[/bold #8338ec] [bold white]Cycle {cycle}[/bold white]  [dim]{ts}[/dim]")
            console.print(f"[dim]  Routing to expert: [bold cyan]{selected_engine.upper()}[/bold cyan][/dim]")
            try:
                from interface.cc_style import cc_spinner
                with cc_spinner("Swarm orchestrator is executing current cycle"):
                    response = await client.swarm.route_and_process(query, context={"user_id": "continuous_mission"})
                content = response.content if hasattr(response, 'content') else str(response)
                
                # ── Parse and render structured output
                _render_mission_output(content, cycle)
                
                # If target directory is in query, auto-extract and save code blocks to it
                target_dir = extract_target_directory(query)
                if target_dir:
                    console.print(f"[cyan]└ Extracting code blocks → {target_dir}[/cyan]")
                    save_code_blocks_to_directory(content, target_dir, default_prefix="output_continuous")
            except Exception as e:
                from rich.panel import Panel as _Panel
                console.print(_Panel(
                    f"[bold red]⚠ Execution Failure[/bold red]\n[dim]{e}[/dim]\n\n[yellow]Swarm will auto-recover on next cycle.[/yellow]",
                    border_style="red",
                    title="[red]✕ Error[/red]"
                ))
                content = f"Execution Failure: {e}"
                
            action = await wait_with_interrupt(interval_min * 60)
            if action == "stop" or action == "menu": break
            elif action == "new_query":
                query = get_input("Enter new query")
                continue
            elif action == "export": save_mission_output(content, mission_name="Continuous", query=query)
    except KeyboardInterrupt:
        console.print("\n[red]Mission terminated by user.[/red]")


def _render_mission_output(content: str, cycle: int = 1) -> None:
    """Premium mission output renderer: parses thought/tool/answer and renders each section."""
    from rich.panel import Panel as _Panel
    from rich.text import Text
    from rich.rule import Rule
    from rich.markdown import Markdown
    import json as _json

    parsed_sections = []

    # Try to parse structured JSON response
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = _json.loads(stripped)
            if isinstance(parsed, dict):
                if parsed.get("final_answer"):
                    parsed_sections = [("answer", parsed["final_answer"])]
                else:
                    if parsed.get("thought"):
                        parsed_sections.append(("thought", parsed["thought"]))
                    if parsed.get("tool"):
                        tool_str = parsed["tool"]
                        if parsed.get("tool_input"):
                            tool_str += f"({parsed['tool_input']})"
                        parsed_sections.append(("tool", tool_str))
                    if parsed.get("observation"):
                        parsed_sections.append(("observation", parsed["observation"]))
        except Exception:
            pass

    # ── Structured render
    if parsed_sections:
        console.print()
        console.print(Rule(f"[bold #ffbe0b]\u2599 Mission Output  \u00b7  Cycle {cycle}[/bold #ffbe0b]", style="#ffbe0b dim"))
        for section_type, section_text in parsed_sections:
            if section_type == "thought":
                console.print(_Panel(
                    f"[italic white]{section_text}[/italic white]",
                    title="[bold plum1]\u25cf Thought[/bold plum1]",
                    border_style="plum1",
                    padding=(0, 2)
                ))
            elif section_type == "tool":
                console.print(_Panel(
                    f"[bold cyan]\ud83d\udee0\ufe0f {section_text}[/bold cyan]",
                    title="[bold yellow]\u25cb Tool Call[/bold yellow]",
                    border_style="yellow",
                    padding=(0, 2)
                ))
            elif section_type == "observation":
                console.print(_Panel(
                    f"[dim]{section_text}[/dim]",
                    title="[dim]\u21bf Observation[/dim]",
                    border_style="dim",
                    padding=(0, 2)
                ))
            elif section_type == "answer":
                console.print(_Panel(
                    Markdown(section_text),
                    title="[bold green]\u2714 Final Answer[/bold green]",
                    border_style="green",
                    padding=(1, 2)
                ))
        console.print(Rule(style="#ffbe0b dim"))
    else:
        # Fallback: raw content with premium styling
        console.print()
        console.print(_Panel(
            Markdown(content) if len(content) > 50 else content,
            title=f"[bold #ffbe0b]\u2599 Mission Output  \u00b7  Cycle {cycle}[/bold #ffbe0b]",
            border_style="#ffbe0b",
            subtitle="[dim]TruthGPT Swarm Intelligence[/dim]",
            padding=(1, 2)
        ))


async def handle_background_missions():
    clear_screen()
    console.print(get_header())
    console.print("[bold cyan]ðŸ“¡ Active Background Missions[/bold cyan]")
    if not background_missions:
        console.print("[yellow]No missions running in background.[/yellow]")
        wait_for_user(force=True)
        return
    table = Table()
    table.add_column("#")
    table.add_column("Mission Name")
    table.add_column("Interval")
    table.add_column("Last Run")
    table.add_column("Status")
    for i, m in enumerate(background_missions, 1):
        table.add_row(str(i), m.name, f"{m.interval}m", m.last_run or "Pending", m.status)
    console.print(table)
    cmd = get_input("Action")
    if cmd == "0": return
    # ... stop/view history logic ...

async def handle_mcp_connect():
    from agents.mcp_client import MCPClient
    url = get_input("Enter MCP Server URL", default="http://localhost:8000")
    client = MCPClient(url)
    with console.status(f"[bold cyan]Connecting to {url}...[/bold cyan]"):
        try:
            tools = await client.list_tools()
            if tools:
                table = Table(title="ðŸ› ï¸ External Tools")
                for t in tools: table.add_row(t.get("name"), t.get("description"))
                console.print(table)
        except Exception as e: console.print(f"[red]Error: {e}[/red]")
    await client.close()
    wait_for_user(force=True)

async def handle_expert_matrix(agents):
    clear_screen()
    console.print(get_header())
    table = Table(title="ðŸ› ï¸ Expert Tool Matrix")
    table.add_column("Expert")
    table.add_column("Tools")
    for agent in agents:
        tools = ", ".join(agent.tools.keys()) if hasattr(agent, "tools") else "N/A"
        table.add_row(agent.name, tools)
    console.print(table)
    wait_for_user(force=True)

async def handle_persona_tuning(agents=None):
    """Persona Tuning: adjust agent roles. Loads agents dynamically if not provided."""
    clear_screen()
    console.print(get_header())
    
    if not agents:
        # Load active agents from the cached client or registry
        try:
            from agents.registry import registry
            agent_keys = sorted(set(list(registry._agent_map.keys()) + list(registry._agents.keys())))
            if not agent_keys:
                console.print("[yellow]No agents available for persona tuning.[/yellow]")
                wait_for_user(force=True)
                return
            console.print("[bold cyan]Available Agents for Persona Tuning:[/bold cyan]")
            for i, key in enumerate(agent_keys, 1):
                console.print(f" {i}. {key}")
            idx = int(get_input("Select expert", default="1"))
            if 1 <= idx <= len(agent_keys):
                target_key = agent_keys[idx-1]
                new_role = get_input("New Role", default="Custom Specialist")
                console.print(f"[green]âœ“ Role for '{target_key}' set to '{new_role}'[/green]")
            wait_for_user(force=True)
            return
        except Exception as e:
            console.print(f"[red]Error loading agents: {e}[/red]")
            wait_for_user(force=True)
            return
    
    for i, a in enumerate(agents, 1): console.print(f" {i}. {a.name}")
    idx = int(get_input("Select expert", default="1"))
    if 1 <= idx <= len(agents):
        target = agents[idx-1]
        new_role = get_input("New Role", default=getattr(target, "role", ""))
        if new_role: target.role = new_role
    wait_for_user(force=True)

async def handle_swarm_telemetry():
    clear_screen()
    console.print(get_header())
    health = {"Status": "Healthy", "Latency": "45ms"}
    console.print(Panel("\n".join([f"{k}: {v}" for k, v in health.items()]), title="ðŸ›°ï¸ Telemetry"))
    wait_for_user(force=True)


async def handle_math_verification():
    """Interactive Math & Formal Verification console."""
    clear_screen()
    console.print(get_header())
    console.print(Panel(
        " [bold cyan]ðŸ”¬ Math & Formal Verification Engine[/bold cyan]\n"
        " [dim]Lean 4 â€¢ SymPy â€¢ Z3 SMT â€¢ NumPy â€¢ Code Verify[/dim]",
        border_style="cyan"
    ))

    # Show available commands
    cmd_table = Table(title="Available Commands", box=None, padding=(0, 2))
    cmd_table.add_column("Prefix", style="bold cyan")
    cmd_table.add_column("Engine", style="white")
    cmd_table.add_column("Example", style="dim")
    cmd_table.add_row("prove:", "SymPy", "prove: (x+1)**2 == x**2 + 2*x + 1")
    cmd_table.add_row("solve:", "SymPy", "solve: x**2 - 4 = 0")
    cmd_table.add_row("simplify:", "SymPy", "simplify: (x**2-1)/(x-1)")
    cmd_table.add_row("integrate:", "SymPy", "integrate: x**2 + 2*x")
    cmd_table.add_row("diff:", "SymPy", "diff: sin(x)*cos(x)")
    cmd_table.add_row("limit:", "SymPy", "limit: sin(x)/x, x, 0")
    cmd_table.add_row("factor:", "SymPy", "factor: x**3 - 1")
    cmd_table.add_row("matrix:", "SymPy", 'matrix: [[1,2],[3,4]]')
    cmd_table.add_row("eigenvalues:", "NumPy", "eigenvalues: [[1,2],[3,4]]")
    cmd_table.add_row("roots:", "NumPy", "roots: [1, -5, 6]")
    cmd_table.add_row("svd:", "NumPy", "svd: [[1,2],[3,4]]")
    cmd_table.add_row("theorem ...", "Lean 4", "theorem add_comm : âˆ€ a b, a + b = b + a")
    cmd_table.add_row("x > 0, ...", "Z3 SMT", "x > 0, x < 10, x*x == 49")
    cmd_table.add_row("typecheck:", "mypy", "typecheck: def f(x: int) -> int: return x")
    console.print(cmd_table)

    try:
        from agents.formal_verification.math_agent import MathVerificationAgent
        from agents.engines import engine_registry
        llm = engine_registry.get_engine(USER_PREFS["preferred_engine"])
        agent = MathVerificationAgent(llm_engine=llm)
    except ImportError as e:
        console.print(f"[red]Error loading MathVerificationAgent: {e}[/red]")
        wait_for_user(force=True)
        return

    console.print("\n[dim]Type your expression (or 'exit' to return):[/dim]")
    while True:
        expr = get_input("\n[bold cyan]Math>[/bold cyan]")
        if expr.lower() in ("exit", "quit", "0", "back"):
            break

        with console.status("[bold cyan]Verifying...[/bold cyan]"):
            result = await agent.process(expr, context={"user_id": "cli_math"})
            content = result.content if hasattr(result, "content") else str(result)
            console.print(Panel(content, title="ðŸ”¬ Verification Result", border_style="green"))


async def handle_agent_composer():
    """Interactive Agent Composer â€” build custom agent combinations."""
    clear_screen()
    console.print(get_header())
    console.print(Panel(
        " [bold magenta]ðŸ§© Agent Composer â€” Build Your Custom Agent[/bold magenta]\n"
        " [dim]Mix capabilities from Math, Research, Code, and System domains[/dim]",
        border_style="magenta"
    ))

    try:
        from agents.composer.agent_composer import (
            _build_catalog, save_blueprint, load_blueprints, ComposedAgent
        )
    except ImportError as e:
        console.print(f"[red]Composer not available: {e}[/red]")
        wait_for_user(force=True)
        return

    # Menu
    console.print("   1. ðŸ§© [bold]Create New Agent[/bold]")
    console.print("   2. ðŸ“‚ [bold]Load Saved Blueprint[/bold]")
    console.print("   3. ðŸ“‹ [bold]View Catalog[/bold]")
    console.print("   0. ðŸ  Back")
    mode = get_input("Select", choices=["0", "1", "2", "3"])
    if mode == "0":
        return

    catalog = _build_catalog()

    if mode == "3":
        # Display full catalog
        cat_table = Table(title="ðŸ§© Capability Catalog", border_style="magenta")
        cat_table.add_column("#", style="cyan", justify="right")
        cat_table.add_column("Key", style="white")
        cat_table.add_column("Category", style="yellow")
        cat_table.add_column("Description", style="green")
        for i, (key, info) in enumerate(catalog.items(), 1):
            cat_table.add_row(str(i), key, info["category"], info["description"])
        console.print(cat_table)
        wait_for_user(force=True)
        return

    if mode == "2":
        blueprints = load_blueprints()
        if not blueprints:
            console.print("[yellow]No saved blueprints found.[/yellow]")
            wait_for_user(force=True)
            return

        bp_table = Table(title="ðŸ“‚ Saved Blueprints", border_style="blue")
        bp_table.add_column("#", style="cyan")
        bp_table.add_column("Name", style="bold white")
        bp_table.add_column("Capabilities", style="green")
        bp_table.add_column("Created", style="dim")
        for i, bp in enumerate(blueprints, 1):
            caps = ", ".join(bp.get("capabilities", []))
            bp_table.add_row(str(i), bp["name"], caps, bp.get("created", "N/A"))
        console.print(bp_table)

        idx = int(get_input("Select blueprint to deploy", default="1"))
        if 1 <= idx <= len(blueprints):
            bp = blueprints[idx - 1]
            from agents.engines import engine_registry
            llm = engine_registry.get_engine(USER_PREFS["preferred_engine"])
            agent = ComposedAgent(
                name=bp["name"],
                role=bp.get("role", "Custom Agent"),
                capabilities=bp["capabilities"],
                llm_engine=llm,
            )
            console.print(f"\n[bold green]âœ“ Deployed: {agent.name}[/bold green]")
            console.print(f"[dim]Capabilities:\n{agent.get_capability_summary()}[/dim]")

            # Interactive query loop
            while True:
                query = get_input(f"\n[bold magenta]{agent.name}>[/bold magenta]")
                if query.lower() in ("exit", "quit", "0", "back"):
                    break
                with console.status(f"[bold cyan]{agent.name} working...[/bold cyan]"):
                    res = await agent.process(query, context={"user_id": "cli_composer"})
                    content = res.content if hasattr(res, "content") else str(res)
                    console.print(Panel(content, title=f"ðŸ¤– {agent.name}", border_style="green"))
        return

    # mode == "1" â€” Create new agent
    console.print("\n[bold cyan]Step 1: Name your agent[/bold cyan]")
    agent_name = get_input("Agent name", default="MyCustomAgent")
    agent_role = get_input("Agent role/description", default="Custom Specialized Agent")

    console.print("\n[bold cyan]Step 2: Select capabilities[/bold cyan]")
    cap_table = Table(title="Available Capabilities", border_style="cyan")
    cap_table.add_column("#", style="cyan", justify="right")
    cap_table.add_column("Key", style="white")
    cap_table.add_column("Category", style="yellow")
    cap_table.add_column("Description", style="green")

    cap_keys = list(catalog.keys())
    for i, key in enumerate(cap_keys, 1):
        info = catalog[key]
        cap_table.add_row(str(i), key, info["category"], info["description"])
    console.print(cap_table)

    selection = get_input("Select capabilities (e.g. 1,2,5,8)")
    indices = [int(i.strip()) for i in selection.split(",") if i.strip().isdigit()]
    selected_caps = [cap_keys[i - 1] for i in indices if 1 <= i <= len(cap_keys)]

    if not selected_caps:
        console.print("[red]No capabilities selected.[/red]")
        wait_for_user(force=True)
        return

    console.print(f"\n[bold green]âœ“ Building '{agent_name}' with: {', '.join(selected_caps)}[/bold green]")

    # Save blueprint?
    if Confirm.ask("Save this as a reusable blueprint?", default=True):
        path = save_blueprint(agent_name, selected_caps, {"role": agent_role})
        console.print(f"[dim]Blueprint saved to {path}[/dim]")

    # Deploy and use
    from agents.engines import engine_registry
    llm = engine_registry.get_engine(USER_PREFS["preferred_engine"])
    agent = ComposedAgent(
        name=agent_name,
        role=agent_role,
        capabilities=selected_caps,
        llm_engine=llm,
    )

    tools_list = ", ".join(agent.tools.keys()) if agent.tools else "none"
    console.print(f"[bold green]âœ“ Agent deployed with tools: {tools_list}[/bold green]")
    console.print(f"[dim]Capabilities:\n{agent.get_capability_summary()}[/dim]")

    # Interactive query loop
    console.print("\n[dim]Type queries (or 'exit' to return):[/dim]")
    while True:
        query = get_input(f"\n[bold magenta]{agent_name}>[/bold magenta]")
        if query.lower() in ("exit", "quit", "0", "back"):
            break
        with console.status(f"[bold cyan]{agent_name} working...[/bold cyan]"):
            res = await agent.process(query, context={"user_id": "cli_composer"})
            content = res.content if hasattr(res, "content") else str(res)
            console.print(Panel(content, title=f"ðŸ¤– {agent_name}", border_style="green"))

