"""
User Input Handling for TruthGPT Interface.

Provides:
- get_input(): Smart input with prompt_toolkit (mouse support) → Rich fallback
- get_choice(): Full-screen interactive menu with mouse/keyboard
- wait_for_user(): Timed pause with auto-continue
- disable_quick_edit(): Windows Console compatibility
"""

import os
import time
from typing import Optional, Dict, List


# ─── Lazy Feature Detection ──────────────────────────────────────────────

_HAS_PROMPT_TOOLKIT: Optional[bool] = None


def _check_prompt_toolkit() -> bool:
    """Check if prompt_toolkit is available (cached)."""
    global _HAS_PROMPT_TOOLKIT
    if _HAS_PROMPT_TOOLKIT is None:
        try:
            import prompt_toolkit  # noqa: F401
            _HAS_PROMPT_TOOLKIT = True
        except ImportError:
            _HAS_PROMPT_TOOLKIT = False
    return _HAS_PROMPT_TOOLKIT


# ─── Windows Console Fix ─────────────────────────────────────────────────

def disable_quick_edit() -> None:
    """Disable QuickEdit mode in Windows Terminal for mouse click capture."""
    if os.name == "nt":
        import ctypes
        try:
            kernel32 = ctypes.windll.kernel32
            h_input = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
            mode = ctypes.c_uint()
            kernel32.GetConsoleMode(h_input, ctypes.byref(mode))

            # Disable QuickEdit (0x0040), Enable Mouse (0x0010),
            # Extended Flags (0x0080), Window Input (0x0008)
            new_mode = (mode.value & ~0x0040) | 0x0010 | 0x0080 | 0x0008
            kernel32.SetConsoleMode(h_input, new_mode)

            # Enable Virtual Terminal Processing for output
            h_output = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            out_mode = ctypes.c_uint()
            kernel32.GetConsoleMode(h_output, ctypes.byref(out_mode))
            kernel32.SetConsoleMode(h_output, out_mode.value | 0x0004)
        except Exception:
            pass


# ─── Primary Input Functions ─────────────────────────────────────────────

def get_input(
    message: str,
    choices: Optional[List[str]] = None,
    default: str = "",
    password: bool = False,
) -> str:
    """Get user input with mouse support if available, otherwise falls back to Rich."""
    from interface.theming import _LAST_ACTIVITY_TIME
    import interface.theming as _theming

    try:
        return _get_input_impl(message, choices, default, password)
    finally:
        _theming._LAST_ACTIVITY_TIME = time.time()


def _get_input_impl(
    message: str,
    choices: Optional[List[str]] = None,
    default: str = "",
    password: bool = False,
) -> str:
    """Internal: attempt prompt_toolkit first, then Rich Prompt."""
    if _check_prompt_toolkit():
        # Check if an event loop is already running to avoid prompt_toolkit crash
        try:
            import asyncio
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False

        if not in_loop:
            try:
                from prompt_toolkit import prompt as pt_prompt
                from prompt_toolkit.styles import Style as PTStyle

                style = PTStyle.from_dict({"prompt": "bold cyan"})
                result = pt_prompt(
                    f"{message}: ", mouse_support=True, style=style, is_password=password
                ).strip()
                if not result and default:
                    return default
                return result
            except (EOFError, KeyboardInterrupt):
                return "0"
            except Exception:
                pass  # Fallback to Rich

    from rich.prompt import Prompt
    return Prompt.ask(message, choices=choices, default=default, password=password)


def wait_for_user(force: bool = False, timeout: int = 3) -> None:
    """Pause with auto-continue after timeout. Respects continuous_mode preference."""
    from interface.preferences import load_user_prefs
    from interface.core import console

    prefs = load_user_prefs()
    if force or not prefs.get("continuous_mode", False):
        try:
            import msvcrt
            console.print(f"\n[dim]Press Enter to continue... (Auto-continuing in {timeout}s)[/dim]", end="")
            start_time = time.time()
            while time.time() - start_time < timeout:
                if msvcrt.kbhit():
                    msvcrt.getch()
                    console.print()
                    return
                time.sleep(0.1)
            console.print("\n[bold yellow]⌛ Idle timeout. Continuing autonomously...[/bold yellow]")
        except ImportError:
            console.print("\n[dim]Auto-continuing...[/dim]")
    else:
        time.sleep(0.1)


# ─── Interactive Choice Menu ─────────────────────────────────────────────

async def get_choice(title: str, options: Dict[str, str], style_name: str = "plum1") -> str:
    """Full-screen interactive choice menu with mouse support."""
    from interface.core import console

    if not _check_prompt_toolkit():
        # Fallback to static print + Prompt.ask
        from rich.table import Table
        from rich.prompt import Prompt

        table = Table(title=title)
        for k, v in options.items():
            table.add_row(k, v)
        console.print(table)
        return Prompt.ask("Select", choices=list(options.keys()))

    from prompt_toolkit.application import Application, get_app
    from prompt_toolkit.layout.containers import HSplit, Window, WindowAlign
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout.layout import Layout
    from prompt_toolkit.widgets import Button, Label, Box, Shadow
    from prompt_toolkit.styles import Style
    from prompt_toolkit.formatted_text import ANSI
    import io

    class SimpleMenuApp:
        def __init__(self):
            self.result = None
            self.kb = KeyBindings()

            @self.kb.add("q")
            @self.kb.add("c-c")
            def _(event):
                event.app.exit()

            # Hotkeys
            for k in options.keys():
                @self.kb.add(k.lower())
                @self.kb.add(k.upper())
                def _(event, val=k):
                    self.result = val
                    event.app.exit()

        def get_layout(self):
            from interface.theming import get_header

            def set_choice(val):
                self.result = val
                get_app().exit(result=val)

            buttons = []
            for k, v in options.items():
                label = f" < {k:>8}: {v:<25} > "
                buttons.append(Button(label, handler=lambda val=k: set_choice(val), width=50))

            # Render header once
            from rich.console import Console as RichConsole
            header_console = RichConsole(file=io.StringIO(), force_terminal=True, width=100)
            header_console.print(get_header())
            header_content = ANSI(header_console.file.getvalue())

            root = HSplit(
                [
                    Window(content=FormattedTextControl(header_content)),
                    Window(height=1),
                    Label(f"  [bold {style_name}] {title.upper()} [/bold {style_name}]", style="bold white"),
                    Window(height=1),
                    HSplit(buttons, padding=1),
                    Window(height=1),
                    Label("   [dim]Click or press key to select[/dim]", style="italic"),
                    Window(height=1),
                ],
                align=WindowAlign.CENTER,
            )

            return Layout(Shadow(Box(root, padding=2)))

        async def run(self):
            pt_style = style_name
            if pt_style == "plum1":
                pt_style = "#ffbbff"
            elif pt_style == "cyan":
                pt_style = "ansicyan"
            elif pt_style == "green":
                pt_style = "ansigreen"
            elif pt_style == "red":
                pt_style = "ansired"

            app = Application(
                layout=self.get_layout(),
                key_bindings=self.kb,
                style=Style.from_dict({"button.focused": f"bg:{pt_style} white"}),
                mouse_support=True,
                full_screen=True,
            )
            await app.run_async()
            return self.result

    app = SimpleMenuApp()
    return await app.run()
