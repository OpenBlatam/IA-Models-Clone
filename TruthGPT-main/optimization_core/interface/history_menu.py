"""
📜 TruthGPT Agent History & Execution Command Center — Platinum Edition
Comprehensive dashboard to monitor, control, and inspect all active, background, and historical agent runs.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.columns import Columns
from rich.align import Align

from interface.core import (
    console, clear_screen, get_header, wait_for_user, background_missions, get_input
)
from interface.cc_style import cc_action, cc_divider, cc_step
from modules.persistence.task_manager import get_persistence_manager, TaskSnapshot

async def agent_history_menu():
    """Main Agent History Dashboard loop."""
    pm = get_persistence_manager()
    
    while True:
        clear_screen()
        console.print(get_header())
        
        console.print("[bold cyan]📜 TRUTHGPT AGENT HISTORY & REAL-TIME MONITORING[/bold cyan]\n")
        
        # Display Stats Summary Panel
        try:
            db_tasks = await pm.list_all_tasks()
        except Exception:
            db_tasks = []
            
        active_db_tasks = [t for t in db_tasks if t.status == "running"]
        paused_db_tasks = [t for t in db_tasks if t.status == "paused"]
        completed_db_tasks = [t for t in db_tasks if t.status == "completed"]
        
        active_bg_missions = background_missions
        
        # Search for trace files on disk
        trace_dir = Path("truthgpt_collected/logs/memory_traces")
        trace_files = []
        if trace_dir.exists():
            trace_files = sorted(list(trace_dir.glob("trace_*.json")), key=lambda p: p.stat().st_mtime, reverse=True)
        
        stats_table = Table.grid(expand=True)
        stats_table.add_column(width=40)
        stats_table.add_column(width=40)
        
        stats_table.add_row(
            Panel(
                f"[bold green]● Active Swarms (Session):[/bold green] [white]{len(active_bg_missions)}[/white]\n"
                f"[bold cyan]● Active DB Tasks:[/bold cyan] [white]{len(active_db_tasks)}[/white]\n"
                f"[bold yellow]● Paused DB Tasks:[/bold yellow] [white]{len(paused_db_tasks)}[/white]",
                title="Active Executions",
                border_style="cyan"
            ),
            Panel(
                f"[bold white]✔ Total Historical Tasks:[/bold white] [green]{len(db_tasks)}[/green]\n"
                f"├─ Completed Runs: [green]{len(completed_db_tasks)}[/green]\n"
                f"└─ Cognitive Trace Logs: [magenta]{len(trace_files)}[/magenta]",
                title="Historical Audits",
                border_style="magenta"
            )
        )
        console.print(stats_table)
        console.print()
        
        # Primary Navigation options
        console.print("   1. 📂 [bold white]Explore All Task History (SQLite Database)[/bold white] [dim]— View completed/active agent runs[/dim]")
        console.print("   2. 📡 [bold white]Session Background Missions[/bold white] [dim]— Manage active background swarms running now[/dim]")
        console.print("   3. 🕵️ [bold white]Cognitive Traces & Decision Logs[/bold white] [dim]— Audit exact multi-agent reasoning steps[/dim]")
        console.print("   0. 🔙 [bold red]Return to Main Dashboard[/bold red]")
        
        choice = get_input("\nSelect historical audit view", choices=["0", "1", "2", "3"], default="1")
        
        if choice == "0":
            break
        elif choice == "1":
            await view_sqlite_tasks(pm)
        elif choice == "2":
            await view_background_missions()
        elif choice == "3":
            await view_cognitive_traces(trace_files)

async def view_sqlite_tasks(pm):
    """Lists and manages all SQLite database task snapshots."""
    while True:
        clear_screen()
        console.print(get_header())
        console.print("[bold cyan]📂 SQLite Database Task Snapshot Logs[/bold cyan]\n")
        
        try:
            tasks = await pm.list_all_tasks()
        except Exception as e:
            console.print(f"[red]Error accessing database: {e}[/red]")
            wait_for_user(force=True)
            return

        # Sort tasks by timestamp desc
        tasks.sort(key=lambda t: t.timestamp, reverse=True)
        
        if not tasks:
            console.print("[yellow]No persisted agent tasks found in database history.[/yellow]")
            wait_for_user(force=True)
            return
            
        table = Table(title="Agent Task History Database", border_style="cyan", expand=True)
        table.add_column("#", justify="right", style="cyan")
        table.add_column("Task ID", style="bold white")
        table.add_column("Agent / Swarm Name", style="magenta")
        table.add_column("Seed Prompt / Task Instruction", style="white", max_width=45)
        table.add_column("Iter", justify="center")
        table.add_column("Updated At", style="dim")
        table.add_column("Status", justify="center")
        
        for i, t in enumerate(tasks, 1):
            status_style = "green" if t.status == "completed" else ("cyan" if t.status == "running" else "yellow")
            status_text = f"[bold {status_style}]{t.status.upper()}[/bold {status_style}]"
            
            # Format timestamp
            t_str = t.timestamp
            try:
                dt = datetime.fromisoformat(t_str)
                t_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass
                
            table.add_row(
                str(i),
                t.task_id[:8],
                t.agent_name,
                t.current_prompt,
                str(t.iteration),
                t_str,
                status_text
            )
            
        console.print(table)
        console.print("\n[dim]Select a task number to inspect details, or '0' to go back.[/dim]")
        
        choice = get_input("Inspect Task #", default="0")
        if choice == "0":
            break
            
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(tasks):
                await inspect_task_details(pm, tasks[idx-1])
            else:
                console.print("[red]Invalid index selection.[/red]")
                time.sleep(0.5)

async def inspect_task_details(pm, task: TaskSnapshot):
    """Detailed inspection panel for a specific task snapshot."""
    while True:
        clear_screen()
        console.print(get_header())
        console.print(f"[bold cyan]🔍 Task Audit Panel: {task.task_id[:8]}[/bold cyan]\n")
        
        # Details Panel
        status_style = "green" if task.status == "completed" else ("cyan" if task.status == "running" else "yellow")
        status_text = f"[bold {status_style}]{task.status.upper()}[/bold {status_style}]"
        
        details = (
            f"[bold white]Task ID:[/bold white] {task.task_id}\n"
            f"[bold white]User ID:[/bold white] {task.user_id}\n"
            f"[bold white]Swarm / Agent Engine:[/bold white] {task.agent_name}\n"
            f"[bold white]Latest Status:[/bold white] {status_text}\n"
            f"[bold white]Iterations Executed:[/bold white] {task.iteration}\n"
            f"[bold white]Timestamp:[/bold white] {task.timestamp}"
        )
        console.print(Panel(details, title="Execution Identity Metadata", border_style="cyan"))
        console.print(Panel(f"[bold white]Prompt Seeds / Objectives:[/bold white]\n{task.current_prompt}", title="Task Goal Instruction", border_style="dim"))
        
        # Render conversation history/thoughts
        if task.history:
            history_table = Table(title="💬 Cognitive Execution & Conversation History", border_style="magenta", expand=True)
            history_table.add_column("Actor/Agent Phase", style="bold cyan", width=25)
            history_table.add_column("Thoughts & Output", style="white")
            
            for msg in task.history:
                role = msg.get("role", "assistant").upper()
                content = msg.get("content", "")
                
                # Check for agent name or specialized sub-agent
                if "agent" in msg:
                    role = f"{role} ({msg['agent']})"
                
                # Color code roles
                role_style = "bold green" if "user" in role.lower() else "bold cyan"
                
                history_table.add_row(
                    f"[{role_style}]{role}[/{role_style}]",
                    content
                )
            console.print(history_table)
        else:
            console.print("[dim italic]No conversation thoughts or historical logs captured for this task yet.[/dim italic]")
            
        console.print("\n[bold cyan]🔧 State Transition Controls[/bold cyan]")
        options = ["0"]
        
        if task.status == "running":
            console.print("   1. ⏸️ [yellow]Pause Execution[/yellow]")
            console.print("   2. 🏁 [green]Mark Completed / Force Terminate[/green]")
            options.extend(["1", "2"])
        elif task.status == "paused":
            console.print("   1. ▶️ [cyan]Resume Execution[/cyan]")
            console.print("   2. 🏁 [green]Mark Completed / Force Terminate[/green]")
            options.extend(["1", "2"])
        elif task.status == "completed":
            console.print("   1. 🔄 [yellow]Re-open & Set Running[/yellow]")
            options.append("1")
            
        console.print("   0. 🔙 Go Back")
        
        choice = get_input("\nAction", choices=options, default="0")
        if choice == "0":
            break
            
        if task.status == "running":
            if choice == "1":
                await pm.update_task_status(task.task_id, "paused")
                task.status = "paused"
                console.print("[yellow]✓ Task paused.[/yellow]")
            elif choice == "2":
                await pm.update_task_status(task.task_id, "completed")
                task.status = "completed"
                console.print("[green]✓ Task marked completed.[/green]")
        elif task.status == "paused":
            if choice == "1":
                await pm.update_task_status(task.task_id, "running")
                task.status = "running"
                console.print("[cyan]✓ Task resumed.[/cyan]")
            elif choice == "2":
                await pm.update_task_status(task.task_id, "completed")
                task.status = "completed"
                console.print("[green]✓ Task marked completed.[/green]")
        elif task.status == "completed":
            if choice == "1":
                await pm.update_task_status(task.task_id, "running")
                task.status = "running"
                console.print("[yellow]✓ Task re-opened to active state.[/yellow]")
                
        time.sleep(0.5)

async def view_background_missions():
    """Lists current session background missions."""
    while True:
        clear_screen()
        console.print(get_header())
        console.print("[bold cyan]📡 Current Session Active Background Swarms[/bold cyan]\n")
        
        active_missions = background_missions
        
        if not active_missions:
            console.print("[yellow]No background missions are running in the current session.[/yellow]\n"
                          "[dim]Launch persistent swarms from option 1 (Swarm Intelligence Hub) -> C (Continuous Mission) or B (Background Missions).[/dim]")
            wait_for_user(force=True)
            return
            
        table = Table(title="Session Background Swarms", border_style="cyan", expand=True)
        table.add_column("#", justify="right", style="cyan")
        table.add_column("Mission Name", style="bold white")
        table.add_column("Interval", justify="center")
        table.add_column("Agents / Team Blueprint", style="magenta")
        table.add_column("Last Execution Time", style="dim")
        table.add_column("Status", justify="center")
        
        for i, m in enumerate(active_missions, 1):
            blueprint = " ➔ ".join(m.team) if hasattr(m, 'team') else "N/A"
            table.add_row(
                str(i),
                m.name,
                f"{m.interval}m",
                blueprint,
                m.last_run or "Pending...",
                f"[bold green]{m.status.upper()}[/bold green]"
            )
            
        console.print(table)
        console.print("\n[dim]Select a mission # to inspect cycles/history, or '0' to go back.[/dim]")
        
        choice = get_input("Inspect Mission #", default="0")
        if choice == "0":
            break
            
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(active_missions):
                await inspect_background_mission(active_missions[idx-1])
            else:
                console.print("[red]Invalid index selection.[/red]")
                time.sleep(0.5)

async def inspect_background_mission(m):
    """Views cycle results and history for a specific active background mission."""
    while True:
        clear_screen()
        console.print(get_header())
        console.print(f"[bold cyan]🔍 Session Background Swarm: {m.name}[/bold cyan]\n")
        
        details = (
            f"[bold white]Mission Seed Prompt:[/bold white] {m.query}\n"
            f"[bold white]Execution Cycle Interval:[/bold white] Every {m.interval} minutes\n"
            f"[bold white]Active Agent Team Blueprint:[/bold white] {' ➔ '.join(m.team)}\n"
            f"[bold white]Status:[/bold white] [bold green]{m.status.upper()}[/bold green]\n"
            f"[bold white]Last Run:[/bold white] {m.last_run or 'Pending...'}"
        )
        console.print(Panel(details, title="Active Swarm Configuration", border_style="cyan"))
        
        # Print run loop history
        if hasattr(m, 'history') and m.history:
            history_table = Table(title="📦 Run Loop Cycle Records & Phase Outputs", border_style="magenta", expand=True)
            history_table.add_column("Cycle Timestamp", style="bold yellow", width=18)
            history_table.add_column("Phase Output / Accomplishments", style="white")
            
            for cycle in m.history:
                timestamp = cycle.get("time", "N/A")
                data = cycle.get("data", [])
                
                cycle_details = []
                for phase in data:
                    agent_name = phase.get("phase", "Unknown")
                    output = phase.get("output", "")
                    
                    # Truncate output for readable summary
                    summary = output[:200] + "..." if len(output) > 200 else output
                    cycle_details.append(f"[bold cyan]● Phase {agent_name}:[/bold cyan] {summary}")
                    
                history_table.add_row(
                    timestamp,
                    "\n\n".join(cycle_details)
                )
            console.print(history_table)
        else:
            console.print("[dim italic]Pending first execution cycle. No history logs generated yet.[/dim italic]")
            
        console.print("\n[bold cyan]🔧 State Transition Controls[/bold cyan]")
        console.print("   1. 🛑 [red]Terminate / Stop Background Swarm[/red]")
        console.print("   0. 🔙 Go Back")
        
        choice = get_input("Action", choices=["0", "1"], default="0")
        if choice == "0":
            break
        elif choice == "1":
            if Confirm.ask("[bold red]Are you sure you want to stop this persistent session swarm?[/bold red]"):
                m.status = "Stopped"
                if hasattr(m, 'task') and m.task:
                    m.task.cancel()
                # Remove from background_missions
                if m in background_missions:
                    background_missions.remove(m)
                console.print("[red]✓ Persistent background swarm stopped and de-registered.[/red]")
                time.sleep(1.0)
                break

async def view_cognitive_traces(trace_files):
    """Lists and inspects cognitive traces / decision logic logs from disk."""
    while True:
        clear_screen()
        console.print(get_header())
        console.print("[bold cyan]🕵️ Agent Cognitive Traces & Multi-Agent Decision Logs[/bold cyan]\n")
        
        if not trace_files:
            console.print("[yellow]No cognitive decision trace files found in logs history.[/yellow]\n"
                          "[dim]Trace logs are generated automatically when running Swarm Blueprints / Dynamic Swarm Fusion.[/dim]")
            wait_for_user(force=True)
            return
            
        table = Table(title="Cognitive Decision Traces", border_style="magenta", expand=True)
        table.add_column("#", justify="right", style="magenta")
        table.add_column("Trace File Name", style="bold white")
        table.add_column("Accrued Date & Time", style="cyan")
        table.add_column("Phases Run", style="white")
        table.add_column("Cumulative Execution Duration", justify="center")
        
        parsed_traces = []
        for i, file_path in enumerate(trace_files, 1):
            try:
                # Load trace details
                with open(file_path, "r", encoding="utf-8") as f:
                    trace_data = json.load(f)
            except:
                trace_data = []
                
            parsed_traces.append((file_path, trace_data))
            
            # Format date from file stats
            mtime = file_path.stat().st_mtime
            dt = datetime.fromtimestamp(mtime)
            dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            
            phases = ", ".join([step.get("phase", "Unknown") for step in trace_data])
            duration_sum = "N/A"
            try:
                durations = [float(step.get("duration", "0").replace("s", "")) for step in trace_data if "duration" in step]
                if durations:
                    duration_sum = f"{sum(durations):.2f}s"
            except:
                pass
                
            table.add_row(
                str(i),
                file_path.name,
                dt_str,
                phases or "Empty Trace",
                duration_sum
            )
            
        console.print(table)
        console.print("\n[dim]Select a trace number to audit the exact decision reasoning, or '0' to go back.[/dim]")
        
        choice = get_input("Inspect Trace #", default="0")
        if choice == "0":
            break
            
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(parsed_traces):
                file_path, trace_data = parsed_traces[idx-1]
                await inspect_trace_steps(file_path.name, trace_data)
            else:
                console.print("[red]Invalid index selection.[/red]")
                time.sleep(0.5)

async def inspect_trace_steps(filename: str, trace_data: List[Dict[str, Any]]):
    """Audits the cognitive reasoning, actions, and rationales for each phase in a trace file."""
    clear_screen()
    console.print(get_header())
    console.print(f"[bold cyan]🕵️ Cognitive Decision Audit: {filename}[/bold cyan]\n")
    
    if not trace_data:
        console.print("[yellow]Empty trace file, no steps to display.[/yellow]")
        wait_for_user(force=True)
        return
        
    for i, step in enumerate(trace_data, 1):
        phase = step.get("phase", "Unknown Agent Phase")
        rationale = step.get("rationale", "No rationale recorded.")
        actions = step.get("actions", [])
        duration = step.get("duration", "N/A")
        speedup = step.get("speedup", "1.0x (Standard)")
        output = step.get("output", "")
        
        actions_str = "\n".join([f"  [bold green]✔[/bold green] [white]{act}[/white]" for act in actions]) if actions else "  No explicit database memory actions recorded."
        
        step_details = (
            f"[bold magenta]🧠 Agent Rationale & Logic Strategy:[/bold magenta]\n{rationale}\n\n"
            f"[bold cyan]🛠️ Actions Executed in Memory/Tools Fabric:[/bold cyan]\n{actions_str}\n\n"
            f"[bold yellow]⏱️ Execution Performance Metrical Load:[/bold yellow]\n"
            f"  └─ Duration: [bold yellow]{duration}[/bold yellow] · Optimistic Speedup: [bold green]{speedup}[/bold green]"
        )
        
        console.print(Panel(step_details, title=f"Phase {i}: Agent [bold magenta]'{phase}'[/bold magenta]", border_style="magenta"))
        
        # Persist full output review
        if output:
            summary = output[:600] + "\n... [dim](output truncated for readability, full file holds complete payload)[/dim]" if len(output) > 600 else output
            console.print(Panel(summary, title=f"🤖 Phase {i}: Result Output Payload Summary", border_style="dim"))
            
        console.print()
        
    wait_for_user(force=True)
