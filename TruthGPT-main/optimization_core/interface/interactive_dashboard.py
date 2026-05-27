import asyncio
import io
import time
import shutil
# rich imports moved inside methods for speed


from rich.console import Console
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import get_app, Application
from prompt_toolkit.layout.containers import VSplit, HSplit, Window, WindowAlign
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.widgets import Button, Box, Shadow, TextArea
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.styles import Style


from interface.core import (
    console as rich_console, USER_PREFS, get_header, get_system_telemetry
)
from interface.cc_style import cc_action, cc_divider


class InteractiveDashboardApp:
    def __init__(self, extended=True):
        self.extended = extended
        self.result = None
        self.kb = KeyBindings()
        self.selected_index = 0
        self._start_time = time.time()
        
        # Create a persistent text area for user command typing
        self.command_input = TextArea(
            multiline=False,
            prompt=' ❯ ',
            style='class:command_input',
            accept_handler=self._on_accept
        )
        
        self._setup_keybindings()
        width = shutil.get_terminal_size().columns or 100
        self.console = Console(file=io.StringIO(), force_terminal=True, width=width)
        
    def _on_accept(self, buffer):
        text = buffer.document.text.strip()
        if text:
            self.result = text
            get_app().exit(result=text)
 
    def _setup_keybindings(self):
        @self.kb.add('c-c')
        def _(event):
            self.result = "exit"
            event.app.exit()
 
        @self.kb.add('escape')
        def _(event):
            self.result = "exit"
            event.app.exit()
 
    def render_rich_content(self):
        from prompt_toolkit.formatted_text import ANSI
        import shutil
        theme = USER_PREFS.get("theme", "industrial")
        
        # Dynamically adapt console width to current terminal size!
        width = shutil.get_terminal_size().columns or 100
        self.console = Console(file=io.StringIO(), force_terminal=True, width=width)
        
        self.console.print(get_header())
        
        if theme in ["claude", "anthropic", "minimalist"]:
            self.console.print()
            return ANSI(self.console.file.getvalue())
 
        self.console.print(f" [dim]System Status: [bold green]ONLINE[/bold green] | Mode: {'Extended' if self.extended else 'Standard'}[/dim]")
        self.console.print("[dim]────────────────────────────────────────────────────────────────────────────────[/dim]")
        
        return ANSI(self.console.file.getvalue())

    def get_layout(self):
        
        theme = USER_PREFS.get("theme", "industrial")
        
        def set_choice(val):
            self.result = val
            get_app().exit(result=val)

        if theme in ["claude", "anthropic", "minimalist"]:
            # Premium claude-style vertical list with section headers
            list_items = []

            # Section divider helper
            def make_section_header(title):
                return Window(
                    content=FormattedTextControl([('class:section_header', f'  ── {title} ')]),
                    height=1,
                    align=WindowAlign.LEFT
                )
            
            def make_claude_item(lid, icon, name, val, index, shortcut_hint=""):
                """Clickable row: ● K  🛡️ Kernel        [K]"""
                def mouse_handler(mouse_event):
                    if mouse_event.event_type == MouseEventType.MOUSE_MOVE:
                        self.selected_index = index
                    elif mouse_event.event_type == MouseEventType.MOUSE_UP:
                        self.selected_index = index
                        set_choice(val)

                hint_part = [('class:shortcut_hint', f'  [{lid}]')] if lid else []
                content = FormattedTextControl(
                    [
                        ('class:menu_bullet', '  ● '),
                        ('class:menu_id',   f'{lid:>2} '),
                        ('class:menu_icon',  f'{icon} '),
                        ('class:menu_name',  f'{name:<22}'),
                    ] + hint_part,
                    focusable=True,
                    show_cursor=False,
                )
                content.mouse_handler = mouse_handler

                return Window(
                    content=content,
                    height=1,
                    cursorline=True,
                    align=WindowAlign.LEFT
                )

            def make_spacer(height=1):
                return Window(height=height, char=" ")

            # ── CORE AGENTS
            list_items.append(make_section_header("CORE AGENTS"))
            list_items.append(make_claude_item("K", "🛡️", "Kernel",   "0", 0))
            list_items.append(make_claude_item("1", "🐝", "Swarm",    "1", 1))
            list_items.append(make_claude_item("2", "🚀", "Frontier",  "2", 2))
            list_items.append(make_claude_item("3", "🔍", "Research",  "3", 3))
            
            if self.extended:
                list_items.append(make_spacer())
                list_items.append(make_section_header("ADVANCED"))
                list_items.append(make_claude_item("4",  "⚙️",  "Opts",         "4",  4))
                list_items.append(make_claude_item("5",  "🧠",  "Labs",         "5",  5))
                list_items.append(make_claude_item("6",  "📱",  "Comm",         "6",  6))
                list_items.append(make_claude_item("9",  "⛓️",  "Web3",         "9",  7))
                list_items.append(make_claude_item("10", "🖥️",  "Node",         "10", 8))
                list_items.append(make_claude_item("11", "📜",  "History",      "11", 9))
                list_items.append(make_claude_item("13", "📊",  "Market",       "13", 10))
                list_items.append(make_claude_item("16", "⚡",  "Overdrive",    "16", 11))
            
            list_items.append(make_spacer())
            list_items.append(make_section_header("SYSTEM"))
            list_items.append(make_claude_item("P",  "👤", "Settings",    "p",    12))
            list_items.append(make_claude_item("99", "➕", "Toggle View", "99",   13))
            list_items.append(make_claude_item("R",  "🔁", "Reboot",      "reboot", 14))
            list_items.append(make_claude_item("X",  "🚪", "Exit",        "exit",  15))
            
            main_content = HSplit(list_items)
        else:
            # Original Industrial Grid
            def make_btn(text, val, width=22):
                return Button(text, handler=lambda: set_choice(val), width=width)

            core_row = VSplit([
                make_btn("K: 🛡️ Kernel", "0"),
                make_btn("1: 🐝 Swarm", "1"),
                make_btn("2: 🚀 Frontier", "2"),
                make_btn("3: 🔍 Research", "3"),
            ], padding=2, align=WindowAlign.CENTER)

            rows = [core_row]
            if self.extended:
                rows.append(VSplit([make_btn("4: ⚙️ Opts", "4"), make_btn("5: 🧠 Labs", "5"), make_btn("6: 📱 Comm", "6"), make_btn("9: ⛓️ Web3", "9")], padding=2, align=WindowAlign.CENTER))
                rows.append(VSplit([make_btn("10: 🖥️ Node", "10"), make_btn("11: 📜 History", "11"), make_btn("13: 📊 Market", "13"), make_btn("15: 🤖 RL", "15")], padding=2, align=WindowAlign.CENTER))
                rows.append(VSplit([make_btn("16: ⚡ Overdrive", "16"), make_btn("P: 👤 Settings", "p"), make_btn("99: ➕ Toggle View", "99"), make_btn("X: 🚪 Exit", "exit")], padding=2, align=WindowAlign.CENTER))
            else:
                rows.append(VSplit([make_btn("99: ➕ Extended View", "99"), make_btn("X: 🚪 Exit", "exit")], padding=2))
            
            main_content = HSplit(rows, padding=1)

        static_content = FormattedTextControl(
            text=self.render_rich_content,
            show_cursor=False,
            modal=True,
        )
        
        # 3. FOOTER (Segmented Status Bar)
        def get_footer_text():
            from interface.telemetry import TelemetryProvider
            tel = get_system_telemetry()
            load_pct = tel["load"]
            mem_pct = tel.get("mem", 0.0)
            filled = int(load_pct / 10)
            load_bar = "█" * filled + "░" * (10 - filled)
            mem_filled = int(mem_pct / 10)
            mem_bar = "█" * mem_filled + "░" * (10 - mem_filled)
            uptime = int(time.time() - self._start_time)
            h, r = divmod(uptime, 3600)
            m, s = divmod(r, 60)
            uptime_str = f"{h:02d}:{m:02d}:{s:02d}"
            
            balances = TelemetryProvider.get_api_balances()
            ds_bal = balances.get("DeepSeek", (0.0, "Est"))
            ds_str = f"${ds_bal[0]:.2f} ({ds_bal[1]})"
            
            return [
                ('class:footer_prompt', ' ❯ INPUT '),
                ('class:footer_sep', '  '),
                ('class:footer_hint', ' ENTER '),
                ('class:footer_label', 'Submit  '),
                ('class:footer_hint', ' ESC '),
                ('class:footer_label', 'Exit   '),
                ('class:footer_sep', '│ '),
                ('class:load_label', f' CPU: {load_bar} {load_pct:.0f}% '),
                ('class:footer_sep', '│ '),
                ('class:mem_label',  f' MEM: {mem_bar} {mem_pct:.0f}% '),
                ('class:footer_sep', '│ '),
                ('class:session_label', f' ⏱ {uptime_str} '),
                ('class:footer_sep', '│ '),
                ('class:session_label', f' 💰 DS: {ds_str} '),
                ('class:footer_sep', '│ '),
                ('class:session_label', f' ⊙ {tel["session_id"]} '),
                ('class:footer_sep', '│ '),
                ('class:version_seg', f' {tel["version"]} '),
            ]
        
        root_container = HSplit([
            Window(content=static_content, wrap_lines=True),
            main_content,
            Window(height=1, char=" "),
            Window(height=1, char="─", style="class:divider"),
            self.command_input,
            Window(height=1, char="─", style="class:divider"),
            Window(height=1, char=" "),
            Window(content=FormattedTextControl(get_footer_text), height=1, align=WindowAlign.LEFT),
        ], align=WindowAlign.LEFT)
        
        if theme in ["claude", "anthropic", "minimalist"]:
            return Layout(root_container, focused_element=self.command_input)
        return Layout(Shadow(Box(root_container, padding=1)), focused_element=self.command_input)

    async def run(self):
        
        theme = USER_PREFS.get("theme", "industrial")
        
        style = Style.from_dict({
            # Menu items
            'menu_bullet':     'bold cyan',
            'menu_id':         'bold #ffbe0b',
            'menu_icon':       '',
            'menu_name':       'white',
            'shortcut_hint':   'dim',
            'section_header':  'dim italic',
            'cursorline':      'bg:#1a1a2e',
            # Button styles (industrial mode)
            'button':          'bold white bg:#0d0d0d',
            'button.focused':  'bold white bg:#8338ec',
            # Footer
            'footer_prompt':   'bg:#8338ec white bold',
            'footer_sep':      'dim',
            'footer_hint':     'bg:#222244 cyan bold',
            'footer_label':    'dim white',
            'load_label':      'dim green',
            'mem_label':       'dim cyan',
            'session_label':   'dim',
            'version_seg':     'dim',
            # Divider
            'divider':         'dim',
            # Command Input
            'command_input':        'bold cyan',
            'command_input.prompt': 'bold #8338ec',
            # Labels
            'label':           'bold cyan',
            'frame.label':     'bold #ffbe0b',
        })
        
        app = Application(
            layout=self.get_layout(),
            key_bindings=self.kb,
            style=style,
            mouse_support=True,
            full_screen=True,
            refresh_interval=1.0,
        )
        await app.run_async()
        return self.result

async def get_dashboard_choice(extended=True):
    app = InteractiveDashboardApp(extended=extended)
    return await app.run()
