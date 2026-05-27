
import asyncio
from prompt_toolkit.application import Application
from prompt_toolkit.layout.containers import VSplit, HSplit, Window, WindowAlign
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.widgets import Frame, Button, Label, Box
from prompt_toolkit.styles import Style

def create_app():
    kb = KeyBindings()

    @kb.add('q')
    @kb.add('c-c')
    def _(event):
        event.app.exit()

    result = {"choice": None}

    def set_choice(val):
        result["choice"] = val
        app.exit()

    # Create buttons for the menu
    buttons = [
        Button("A: Ask Swarm", handler=lambda: set_choice("A")),
        Button("C: Continuous", handler=lambda: set_choice("C")),
        Button("P: Persona", handler=lambda: set_choice("P")),
        Button("V: Neural Vault", handler=lambda: set_choice("V")),
        Button("S: Status", handler=lambda: set_choice("S")),
        Button("X: Composer", handler=lambda: set_choice("X")),
        Button("F: Fusion", handler=lambda: set_choice("F")),
        Button("B: Background", handler=lambda: set_choice("B")),
        Button("E: Matrix", handler=lambda: set_choice("E")),
        Button("M: MCP", handler=lambda: set_choice("M")),
        Button("T: Math", handler=lambda: set_choice("T")),
        Button("0: Back", handler=lambda: set_choice("0")),
    ]

    # Arrange buttons in a grid (2 columns)
    left_col = HSplit(buttons[:6], padding=1)
    right_col = HSplit(buttons[6:], padding=1)
    
    root_container = Box(
        Frame(
            HSplit([
                Label("Swarm Intelligence Hub - Industrial Command Center", style="bold magenta"),
                VSplit([left_col, right_col], padding=4),
            ], align=WindowAlign.CENTER),
            title="Industrial Command Center"
        )
    )

    style = Style.from_dict({
        'frame.label': 'bold magenta',
        'button.focused': 'bg:magenta white',
        'button': 'white',
    })

    app = Application(
        layout=Layout(root_container),
        key_bindings=kb,
        style=style,
        mouse_support=True,
        full_screen=True
    )
    return app, result

async def main():
    app, result = create_app()
    await app.run_async()
    print(f"Selected: {result['choice']}")

if __name__ == "__main__":
    asyncio.run(main())
