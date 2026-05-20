from rich.layout import Layout
from rich.text import Text
# ... en la parte final, reemplazar:
# console.print(Panel(result, ...))
# por:
layout = Layout()
layout.split_column(
    Layout(Panel(Text("TruthGPT", style="bold cyan"), border_style="green"), name="header", size=3),
    Layout(Panel(result, title="[bold]Resultado[/bold]", border_style="green"), name="output"),
    Layout(Table("Métrica", "Valor", title="Estadísticas"), name="stats", size=5)
)
console.print(layout)