"""
Ejemplo de uso del Physical Store Designer AI
=============================================

Este script demuestra cómo usar el servicio de diseño de tiendas físicas
para generar diseños completos incluyendo:
- Plan de marketing
- Plan de decoración
- Visualizaciones
- Análisis financiero

Configuración mediante variables de entorno:
- PSD_SECTION_WIDTH: Ancho de secciones (default: 60)
- PSD_MAX_STRATEGY_LENGTH: Longitud máxima de estrategias (default: 80)
- PSD_MAX_FURNITURE_LENGTH: Longitud máxima de muebles (default: 70)
- PSD_MAX_PROMPT_LENGTH: Longitud máxima de prompts (default: 60)
- PSD_MAX_DESCRIPTION_LENGTH: Longitud máxima de descripción (default: 500)
- PSD_MAX_ITEMS_TO_SHOW: Máximo de items a mostrar (default: 5)
- PSD_MAX_STRATEGIES_TO_SHOW: Máximo de estrategias a mostrar (default: 3)
- PSD_EXPORT_DIR: Directorio de exportación (default: exports)
- PSD_MAX_RETRIES: Máximo de reintentos (default: 3)
- PSD_RETRY_DELAY: Delay entre reintentos en segundos (default: 1.0)
"""

import asyncio
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Any, Dict, List, Generator

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import arrow
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False
    from datetime import datetime

try:
    from slugify import slugify
    SLUGIFY_AVAILABLE = True
except ImportError:
    SLUGIFY_AVAILABLE = False

from .core.models import (
    StoreDesignRequest,
    StoreDesign,
    StoreType,
    DesignStyle
)
from .services.store_designer_service import StoreDesignerService
from .core.exceptions import PhysicalStoreDesignerError
from .core.logging_config import get_logger, setup_logging
from .core.utils.format_utils import format_bytes

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Inicializar Rich console si está disponible
console = Console() if RICH_AVAILABLE else None

# Constantes de formato (configurables mediante variables de entorno)
SECTION_WIDTH = int(os.getenv("PSD_SECTION_WIDTH", "60"))
MAX_STRATEGY_LENGTH = int(os.getenv("PSD_MAX_STRATEGY_LENGTH", "80"))
MAX_FURNITURE_LENGTH = int(os.getenv("PSD_MAX_FURNITURE_LENGTH", "70"))
MAX_PROMPT_LENGTH = int(os.getenv("PSD_MAX_PROMPT_LENGTH", "60"))
MAX_DESCRIPTION_LENGTH = int(os.getenv("PSD_MAX_DESCRIPTION_LENGTH", "500"))
MAX_ITEMS_TO_SHOW = int(os.getenv("PSD_MAX_ITEMS_TO_SHOW", "5"))
MAX_STRATEGIES_TO_SHOW = int(os.getenv("PSD_MAX_STRATEGIES_TO_SHOW", "3"))
DEFAULT_EXPORT_DIR = Path(os.getenv("PSD_EXPORT_DIR", "exports"))
MAX_RETRIES = int(os.getenv("PSD_MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("PSD_RETRY_DELAY", "1.0"))


def _format_datetime(dt: Any) -> str:
    """Formatear datetime usando arrow si está disponible."""
    if ARROW_AVAILABLE and hasattr(dt, 'isoformat'):
        try:
            return arrow.get(dt).humanize(locale='es')
        except Exception:
            pass
    
    if isinstance(dt, str):
        return dt
    
    if hasattr(dt, 'strftime'):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    return str(dt)


def _slugify_text(text: str) -> str:
    """Crear slug del texto usando python-slugify si está disponible."""
    if SLUGIFY_AVAILABLE:
        return slugify(text)
    # Fallback simple
    return text.lower().replace(' ', '_').replace('/', '_')


@contextmanager
def timing_context() -> Generator[float, None, None]:
    """Context manager para medir tiempo de ejecución.
    
    Yields:
        Tiempo de inicio
    """
    start = time.time()
    yield start


def _truncate_text(text: str, max_length: int) -> str:
    """Truncar texto si excede longitud máxima."""
    text_str = str(text)
    if len(text_str) > max_length:
        return f"{text_str[:max_length]}..."
    return text_str


def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """Obtener atributo de forma segura."""
    return getattr(obj, attr, default) if hasattr(obj, attr) else default


def _format_list_items(items: List[Any], max_items: int, max_length: int) -> None:
    """Formatear lista de items con truncamiento usando Rich si está disponible."""
    for i, item in enumerate(items[:max_items], 1):
        item_str = str(item) if not isinstance(item, dict) else item.get('name', str(item))
        truncated = _truncate_text(item_str, max_length)
        if RICH_AVAILABLE and console:
            console.print(f"    [cyan]{i}.[/cyan] {truncated}")
        else:
            print(f"    {i}. {truncated}")


def _validate_dimensions(dimensions: Dict[str, Any]) -> bool:
    """Validar que las dimensiones sean correctas."""
    if not isinstance(dimensions, dict):
        return False
    required = {'width', 'length', 'height'}
    return all(key in dimensions and isinstance(dimensions[key], (int, float)) 
               and dimensions[key] > 0 for key in required)


def _validate_store_name(store_name: str) -> bool:
    """Validar nombre de tienda."""
    return bool(store_name and len(store_name.strip()) >= 3)


def print_section(title: str, char: str = "=") -> None:
    """Imprimir sección con formato usando Rich si está disponible."""
    if RICH_AVAILABLE and console:
        console.print(f"\n[bold blue]{char * SECTION_WIDTH}[/bold blue]")
        console.print(f"  [bold]{title}[/bold]")
        console.print(f"[bold blue]{char * SECTION_WIDTH}[/bold blue]\n")
    else:
        print(f"\n{char * SECTION_WIDTH}")
        print(f"  {title}")
        print(f"{char * SECTION_WIDTH}\n")


def print_field(label: str, value: Any, indent: int = 2) -> None:
    """Imprimir campo con formato consistente usando Rich si está disponible."""
    spaces = " " * indent
    if isinstance(value, (list, dict)):
        value_str = f"{len(value)} elementos" if isinstance(value, list) else f"{len(value)} campos"
    else:
        value_str = str(value)
    
    if RICH_AVAILABLE and console:
        console.print(f"{spaces}[green]•[/green] [bold]{label}:[/bold] {value_str}")
    else:
        print(f"{spaces}• {label}: {value_str}")


def format_currency(amount: float) -> str:
    """Formatear cantidad como moneda."""
    return f"${amount:,.2f}"


def _format_dimensions(dims: Dict[str, Any]) -> str:
    """Formatear dimensiones."""
    width = dims.get('width', 'N/A')
    length = dims.get('length', 'N/A')
    height = dims.get('height', 'N/A')
    return f"{width}m × {length}m × {height}m"


def display_design_summary(design: StoreDesign) -> None:
    """Mostrar resumen del diseño generado usando Rich Table si está disponible.
    
    Args:
        design: Diseño de tienda a mostrar
    """
    logger.debug("Mostrando resumen del diseño", extra={"store_id": getattr(design, 'store_id', None)})
    
    if RICH_AVAILABLE and console:
        table = Table(title="✅ Diseño Generado", box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Campo", style="cyan", no_wrap=True)
        table.add_column("Valor", style="green")
        
        table.add_row("Nombre", design.store_name)
        table.add_row("Tipo", design.store_type.value)
        table.add_row("Estilo", design.style.value if design.style else "No especificado")
        
        layout = _safe_getattr(design, 'layout')
        if layout:
            dims = _safe_getattr(layout, 'dimensions')
            if isinstance(dims, dict):
                table.add_row("Dimensiones", _format_dimensions(dims))
        
        console.print(table)
    else:
        print_section("✅ Diseño Generado", "=")
        print_field("Nombre", design.store_name)
        print_field("Tipo", design.store_type.value)
        print_field("Estilo", design.style.value if design.style else "No especificado")
        
        layout = _safe_getattr(design, 'layout')
        if layout:
            dims = _safe_getattr(layout, 'dimensions')
            if isinstance(dims, dict):
                print_field("Dimensiones", _format_dimensions(dims))


def display_marketing_plan(design: StoreDesign) -> None:
    """Mostrar plan de marketing.
    
    Args:
        design: Diseño de tienda con plan de marketing
    """
    marketing_plan = _safe_getattr(design, 'marketing_plan')
    if not marketing_plan:
        return
    
    print_section("📊 Plan de Marketing", "-")
    
    target_audience = _safe_getattr(marketing_plan, 'target_audience')
    if target_audience:
        print_field("Audiencia Objetivo", target_audience)
    
    strategies = _safe_getattr(marketing_plan, 'marketing_strategy')
    if isinstance(strategies, list) and strategies:
        print_field("Estrategias de Marketing", strategies)
        _format_list_items(strategies, MAX_STRATEGIES_TO_SHOW, MAX_STRATEGY_LENGTH)
    
    tactics = _safe_getattr(marketing_plan, 'sales_tactics')
    if isinstance(tactics, list) and tactics:
        print_field("Tácticas de Ventas", tactics)
        _format_list_items(tactics, MAX_STRATEGIES_TO_SHOW, MAX_STRATEGY_LENGTH)


def display_decoration_plan(design: StoreDesign) -> None:
    """Mostrar plan de decoración.
    
    Args:
        design: Diseño de tienda con plan de decoración
    """
    decoration_plan = _safe_getattr(design, 'decoration_plan')
    if not decoration_plan:
        return
    
    print_section("🎨 Plan de Decoración", "-")
    
    color_scheme = _safe_getattr(decoration_plan, 'color_scheme')
    if color_scheme:
        print_field("Esquema de Colores", color_scheme)
    
    furniture = _safe_getattr(decoration_plan, 'furniture_recommendations')
    if isinstance(furniture, list) and furniture:
        print_field("Recomendaciones de Muebles", furniture)
        _format_list_items(furniture, MAX_ITEMS_TO_SHOW, MAX_FURNITURE_LENGTH)
    
    elements = _safe_getattr(decoration_plan, 'decoration_elements')
    if elements:
        print_field("Elementos Decorativos", elements)
    
    budget = _safe_getattr(decoration_plan, 'budget_estimate')
    if isinstance(budget, dict) and budget:
        total = sum(budget.values())
        if total > 0:
            print_field("Presupuesto Estimado Total", format_currency(total))
            if RICH_AVAILABLE and console:
                console.print("    [dim]Desglose:[/dim]")
                for category, amount in list(budget.items())[:MAX_ITEMS_TO_SHOW]:
                    console.print(f"      [yellow]-[/yellow] {category}: [green]{format_currency(amount)}[/green]")
            else:
                print("    Desglose:")
                for category, amount in list(budget.items())[:MAX_ITEMS_TO_SHOW]:
                    print(f"      - {category}: {format_currency(amount)}")


def display_visualizations(design: StoreDesign) -> None:
    """Mostrar visualizaciones generadas.
    
    Args:
        design: Diseño de tienda con visualizaciones
    """
    visualizations = _safe_getattr(design, 'visualizations')
    if not visualizations or not isinstance(visualizations, list):
        return
    
    print_section("🖼️ Visualizaciones", "-")
    print_field("Total de Visualizaciones", visualizations)
    
    for i, viz in enumerate(visualizations[:MAX_ITEMS_TO_SHOW], 1):
        view_type = _safe_getattr(viz, 'view_type', 'N/A')
        prompt = _safe_getattr(viz, 'image_prompt', 'N/A')
        prompt_preview = _truncate_text(str(prompt), MAX_PROMPT_LENGTH)
        if RICH_AVAILABLE and console:
            console.print(f"    [cyan]{i}.[/cyan] [bold]{view_type}:[/bold] {prompt_preview}")
        else:
            print(f"    {i}. {view_type}: {prompt_preview}")


def display_description(design: StoreDesign) -> None:
    """Mostrar descripción del diseño usando Rich Panel si está disponible.
    
    Args:
        design: Diseño de tienda con descripción
    """
    description = _safe_getattr(design, 'description')
    if not description:
        return
    
    if isinstance(description, str):
        if RICH_AVAILABLE and console:
            if len(description) > MAX_DESCRIPTION_LENGTH:
                content = f"{description[:MAX_DESCRIPTION_LENGTH]}...\n\n[dim](Descripción truncada, {len(description)} caracteres totales)[/dim]"
            else:
                content = description
            
            panel = Panel(
                content,
                title="📝 Descripción del Diseño",
                border_style="blue",
                box=box.ROUNDED
            )
            console.print(panel)
        else:
            print_section("📝 Descripción del Diseño", "-")
            if len(description) > MAX_DESCRIPTION_LENGTH:
                print(f"    {description[:MAX_DESCRIPTION_LENGTH]}...")
                print(f"    (Descripción truncada, {len(description)} caracteres totales)")
            else:
                paragraphs = description.split('\n')
                for para in paragraphs:
                    if para.strip():
                        print(f"    {para.strip()}")
    else:
        print_section("📝 Descripción del Diseño", "-")
        print_field("Descripción", description)


def _display_performance_info(elapsed_time: float) -> None:
    """Mostrar información de rendimiento usando Rich Panel si está disponible."""
    if RICH_AVAILABLE and console:
        panel = Panel(
            f"[bold green]Tiempo de generación:[/bold green] {elapsed_time:.2f} segundos",
            title="⏱️ Información de Rendimiento",
            border_style="yellow",
            box=box.ROUNDED
        )
        console.print(panel)
    else:
        print_section("⏱️ Información de Rendimiento", "-")
        print(f"    Tiempo de generación: {elapsed_time:.2f} segundos")
    logger.info("Diseño generado", extra={"elapsed_time": elapsed_time})


def _display_statistics(design: StoreDesign) -> None:
    """Mostrar estadísticas resumidas del diseño."""
    stats = []
    
    marketing_plan = _safe_getattr(design, 'marketing_plan')
    if marketing_plan:
        strategies = _safe_getattr(marketing_plan, 'marketing_strategy', [])
        tactics = _safe_getattr(marketing_plan, 'sales_tactics', [])
        if isinstance(strategies, list):
            stats.append(f"Estrategias: {len(strategies)}")
        if isinstance(tactics, list):
            stats.append(f"Tácticas: {len(tactics)}")
    
    decoration_plan = _safe_getattr(design, 'decoration_plan')
    if decoration_plan:
        furniture = _safe_getattr(decoration_plan, 'furniture_recommendations', [])
        if isinstance(furniture, list):
            stats.append(f"Muebles: {len(furniture)}")
    
    visualizations = _safe_getattr(design, 'visualizations', [])
    if isinstance(visualizations, list):
        stats.append(f"Visualizaciones: {len(visualizations)}")
    
    if stats:
        print_section("📈 Estadísticas", "-")
        if RICH_AVAILABLE and console:
            console.print("    " + " | ".join(stats))
        else:
            print("    " + " | ".join(stats))


def _display_all_sections(design: StoreDesign, show_stats: bool = True) -> None:
    """Mostrar todas las secciones del diseño."""
    display_design_summary(design)
    display_description(design)
    display_marketing_plan(design)
    display_decoration_plan(design)
    display_visualizations(design)
    if show_stats:
        _display_statistics(design)


def export_design_to_json(design: StoreDesign, output_path: Optional[Path] = None) -> Path:
    """Exportar diseño a archivo JSON.
    
    Args:
        design: Diseño a exportar
        output_path: Ruta de salida (opcional)
        
    Returns:
        Ruta del archivo exportado
        
    Raises:
        IOError: Si hay error al escribir el archivo
    """
    if output_path is None:
        DEFAULT_EXPORT_DIR.mkdir(exist_ok=True)
        store_id = _safe_getattr(design, 'store_id', 'unknown')
        store_name_slug = _slugify_text(design.store_name)
        if ARROW_AVAILABLE:
            timestamp = arrow.now().format('YYYYMMDD_HHmmss')
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_EXPORT_DIR / f"design_{store_name_slug}_{timestamp}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    design_dict = design.dict() if hasattr(design, 'dict') else design.model_dump() if hasattr(design, 'model_dump') else {}
    
    # Agregar metadata de exportación
    if ARROW_AVAILABLE:
        exported_at = arrow.now().isoformat()
    else:
        from datetime import datetime
        exported_at = datetime.now().isoformat()
    
    export_data = {
        "exported_at": exported_at,
        "version": "1.0",
        "design": design_dict
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        # Verificar que el archivo se escribió correctamente
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise IOError(f"El archivo exportado está vacío o no existe: {output_path}")
        
        logger.info("Diseño exportado", extra={
            "output_path": str(output_path),
            "file_size": output_path.stat().st_size
        })
        return output_path
    except Exception as e:
        logger.error("Error al exportar diseño", extra={
            "output_path": str(output_path),
            "error": str(e)
        }, exc_info=True)
        raise IOError(f"Error al exportar diseño: {e}") from e


def load_design_from_json(file_path: Path) -> Optional[Dict[str, Any]]:
    """Cargar diseño desde archivo JSON.
    
    Args:
        file_path: Ruta del archivo JSON
        
    Returns:
        Diccionario con los datos del diseño o None si hay error
    """
    if not file_path.exists():
        logger.error("Archivo no encontrado", extra={"file_path": str(file_path)})
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Manejar formato nuevo con metadata o formato antiguo directo
        if isinstance(data, dict) and 'design' in data:
            design_data = data['design']
            logger.info("Diseño cargado con metadata", extra={
                "file_path": str(file_path),
                "exported_at": data.get('exported_at')
            })
        else:
            design_data = data
        
        logger.info("Diseño cargado exitosamente", extra={"file_path": str(file_path)})
        return design_data
    except json.JSONDecodeError as e:
        logger.error("Error al parsear JSON", extra={
            "file_path": str(file_path),
            "error": str(e)
        }, exc_info=True)
        return None
    except Exception as e:
        logger.exception("Error al cargar diseño", extra={"file_path": str(file_path)})
        return None


def validate_exported_file(file_path: Path) -> bool:
    """Validar que un archivo exportado sea válido.
    
    Args:
        file_path: Ruta del archivo a validar
        
    Returns:
        True si el archivo es válido, False en caso contrario
    """
    if not file_path.exists():
        logger.warning("Archivo no existe", extra={"file_path": str(file_path)})
        return False
    
    if file_path.stat().st_size == 0:
        logger.warning("Archivo vacío", extra={"file_path": str(file_path)})
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Verificar estructura básica
        if not isinstance(data, dict):
            logger.warning("Archivo no contiene objeto JSON válido", extra={"file_path": str(file_path)})
            return False
        
        # Verificar que tenga diseño o sea formato antiguo
        if 'design' not in data and 'store_name' not in data:
            logger.warning("Archivo no contiene datos de diseño válidos", extra={"file_path": str(file_path)})
            return False
        
        logger.info("Archivo validado exitosamente", extra={"file_path": str(file_path)})
        return True
    except json.JSONDecodeError:
        logger.error("Archivo no es JSON válido", extra={"file_path": str(file_path)})
        return False
    except Exception as e:
        logger.exception("Error al validar archivo", extra={"file_path": str(file_path)})
        return False


async def _generate_design_with_retry(
    service: StoreDesignerService,
    request: StoreDesignRequest,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY
) -> StoreDesign:
    """Generar diseño con retry logic y exponential backoff.
    
    Args:
        service: Servicio de diseño
        request: Request de diseño
        max_retries: Número máximo de reintentos
        retry_delay: Delay inicial entre reintentos
        
    Returns:
        Diseño generado
        
    Raises:
        PhysicalStoreDesignerError: Si falla después de todos los reintentos
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return await service.generate_store_design(request)
        except (PhysicalStoreDesignerError, Exception) as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    f"Intento {attempt + 1}/{max_retries} falló, reintentando en {delay:.2f}s",
                    extra={
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "error": str(e),
                        "delay": delay
                    }
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"Todos los intentos fallaron después de {max_retries} reintentos",
                    extra={"max_retries": max_retries},
                    exc_info=True
                )
    
    raise last_error


async def generate_store_design_example(
    store_name: str,
    store_type: StoreType,
    style: Optional[DesignStyle] = None,
    show_timing: bool = True,
    show_stats: bool = True,
    export: bool = False,
    export_path: Optional[Path] = None,
    max_retries: Optional[int] = None,
    **kwargs: Any
) -> Optional[StoreDesign]:
    """Generar y mostrar diseño de tienda.
    
    Args:
        store_name: Nombre de la tienda
        store_type: Tipo de tienda
        style: Estilo de diseño opcional
        show_timing: Si mostrar información de tiempo de ejecución
        show_stats: Si mostrar estadísticas resumidas
        export: Si exportar diseño a JSON
        export_path: Ruta personalizada para exportación
        max_retries: Número máximo de reintentos (usa PSD_MAX_RETRIES si None)
        **kwargs: Argumentos adicionales para StoreDesignRequest
        
    Returns:
        El diseño generado o None si hay error
    """
    try:
        # Validar entrada
        if not _validate_store_name(store_name):
            logger.error("Nombre de tienda inválido", extra={"store_name": store_name})
            if RICH_AVAILABLE and console:
                console.print("[bold red]❌ Error:[/bold red] El nombre de la tienda debe tener al menos 3 caracteres")
            else:
                print("❌ Error: El nombre de la tienda debe tener al menos 3 caracteres", file=sys.stderr)
            return None
        
        logger.info("Iniciando generación de diseño", extra={
            "store_name": store_name,
            "store_type": store_type.value,
            "style": style.value if style else None
        })
        
        # Validar dimensiones si están presentes
        dimensions = kwargs.get('dimensions')
        if dimensions and not _validate_dimensions(dimensions):
            logger.warning("Dimensiones inválidas detectadas", extra={"dimensions": dimensions})
            if RICH_AVAILABLE and console:
                console.print("[yellow]⚠️  Advertencia: Dimensiones inválidas, usando valores por defecto[/yellow]")
            else:
                print("⚠️  Advertencia: Dimensiones inválidas, usando valores por defecto", file=sys.stderr)
            kwargs.pop('dimensions', None)
        
        service = StoreDesignerService()
        
        request = StoreDesignRequest(
            store_name=store_name,
            store_type=store_type,
            style_preference=style,
            **kwargs
        )
        
        retries = max_retries if max_retries is not None else MAX_RETRIES
        
        if RICH_AVAILABLE and console:
            console.print(f"\n[bold blue]🏪 Generando diseño para:[/bold blue] [cyan]{store_name}[/cyan]")
            if retries > 1:
                console.print(f"[dim]Reintentos configurados: {retries}[/dim]")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("⏳ Por favor espere...", total=None)
                with timing_context() as start:
                    design = await _generate_design_with_retry(service, request, max_retries=retries)
                    elapsed_time = time.time() - start
                    progress.update(task, completed=True)
        else:
            print(f"\n🏪 Generando diseño para: {store_name}")
            if retries > 1:
                print(f"   (Reintentos configurados: {retries})")
            print("⏳ Por favor espere...\n")
            with timing_context() as start:
                design = await _generate_design_with_retry(service, request, max_retries=retries)
                elapsed_time = time.time() - start
        
        logger.info("Diseño generado exitosamente", extra={
            "store_id": getattr(design, 'store_id', None),
            "elapsed_time": elapsed_time
        })
        
        _display_all_sections(design, show_stats=show_stats)
        
        if show_timing:
            _display_performance_info(elapsed_time)
        
        if export:
            try:
                exported_path = export_design_to_json(design, export_path)
                file_size = exported_path.stat().st_size
                if RICH_AVAILABLE and console:
                    console.print(f"\n[green]💾 Diseño exportado a:[/green] [cyan]{exported_path}[/cyan]")
                    console.print(f"   [dim]Tamaño del archivo:[/dim] {format_bytes(file_size)}")
                else:
                    print(f"\n💾 Diseño exportado a: {exported_path}")
                    print(f"   Tamaño del archivo: {format_bytes(file_size)}")
            except IOError as e:
                logger.error("Error al exportar diseño", extra={"error": str(e)})
                if RICH_AVAILABLE and console:
                    console.print(f"[yellow]⚠️  Advertencia: No se pudo exportar el diseño:[/yellow] {e}")
                else:
                    print(f"\n⚠️  Advertencia: No se pudo exportar el diseño: {e}", file=sys.stderr)
        
        if RICH_AVAILABLE and console:
            success_panel = Panel(
                "[bold green]✅ Proceso completado exitosamente[/bold green]",
                box=box.ROUNDED,
                border_style="green"
            )
            console.print(success_panel)
        else:
            print("\n" + "=" * SECTION_WIDTH)
            print("✅ Proceso completado exitosamente")
            print("=" * SECTION_WIDTH + "\n")
        
        return design
        
    except PhysicalStoreDesignerError as e:
        logger.error("Error del servicio al generar diseño", extra={
            "error_type": type(e).__name__,
            "error_message": str(e)
        }, exc_info=True)
        if RICH_AVAILABLE and console:
            error_panel = Panel(
                f"[bold red]❌ Error del servicio:[/bold red] {e}",
                title="Error",
                border_style="red",
                box=box.ROUNDED
            )
            console.print(error_panel)
        else:
            print(f"\n❌ Error del servicio: {e}", file=sys.stderr)
        return None
    except Exception as e:
        logger.exception("Error inesperado al generar diseño", extra={
            "error_type": type(e).__name__
        })
        if RICH_AVAILABLE and console:
            error_panel = Panel(
                f"[bold red]❌ Error inesperado:[/bold red] {type(e).__name__}: {e}",
                title="Error",
                border_style="red",
                box=box.ROUNDED
            )
            console.print(error_panel)
        else:
            print(f"\n❌ Error inesperado: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _print_message(message: str, msg_type: str = "info") -> None:
    """Imprimir mensaje usando Rich si está disponible.
    
    Args:
        message: Mensaje a imprimir
        msg_type: Tipo de mensaje (info, success, warning, error)
    """
    if RICH_AVAILABLE and console:
        colors = {
            "info": "blue",
            "success": "green",
            "warning": "yellow",
            "error": "red"
        }
        color = colors.get(msg_type, "white")
        console.print(f"[{color}]{message}[/{color}]")
    else:
        print(message)


def _display_design_metadata(design: StoreDesign) -> None:
    """Mostrar metadata adicional del diseño."""
    metadata_items = []
    
    store_id = _safe_getattr(design, 'store_id')
    if store_id:
        metadata_items.append(f"💾 ID: {store_id}")
    
    created_at = _safe_getattr(design, 'created_at')
    if created_at:
        date_str = _format_datetime(created_at)
        metadata_items.append(f"📅 Creado: {date_str}")
    
    if metadata_items:
        print_section("📋 Metadata", "-")
        if RICH_AVAILABLE and console:
            for item in metadata_items:
                console.print(f"    {item}")
        else:
            for item in metadata_items:
                print(f"    {item}")


def compare_designs(design1: StoreDesign, design2: StoreDesign) -> Dict[str, Any]:
    """Comparar dos diseños y retornar diferencias.
    
    Args:
        design1: Primer diseño
        design2: Segundo diseño
        
    Returns:
        Diccionario con comparación de campos
    """
    comparison = {
        "store_name": {
            "design1": design1.store_name,
            "design2": design2.store_name,
            "different": design1.store_name != design2.store_name
        },
        "store_type": {
            "design1": design1.store_type.value,
            "design2": design2.store_type.value,
            "different": design1.store_type != design2.store_type
        },
        "style": {
            "design1": design1.style.value if design1.style else None,
            "design2": design2.style.value if design2.style else None,
            "different": design1.style != design2.style
        }
    }
    
    # Comparar planes de marketing
    mp1 = _safe_getattr(design1, 'marketing_plan')
    mp2 = _safe_getattr(design2, 'marketing_plan')
    if mp1 or mp2:
        strategies1 = _safe_getattr(mp1, 'marketing_strategy', []) if mp1 else []
        strategies2 = _safe_getattr(mp2, 'marketing_strategy', []) if mp2 else []
        comparison["marketing_strategies"] = {
            "design1_count": len(strategies1) if isinstance(strategies1, list) else 0,
            "design2_count": len(strategies2) if isinstance(strategies2, list) else 0,
            "different": len(strategies1) != len(strategies2) if isinstance(strategies1, list) and isinstance(strategies2, list) else True
        }
    
    logger.info("Diseños comparados", extra={
        "design1_id": getattr(design1, 'store_id', None),
        "design2_id": getattr(design2, 'store_id', None)
    })
    
    return comparison


async def main() -> None:
    """Función principal con ejemplo de uso.
    
    Puede generar múltiples diseños y compararlos.
    """
    logger.info("Iniciando ejemplo de uso")
    
    # Ejemplo 1: Café moderno
    design1 = await generate_store_design_example(
        store_name="Café Moderno del Centro",
        store_type=StoreType.CAFE,
        style=DesignStyle.MODERN,
        budget_range="medio",
        location="Centro de la ciudad, zona comercial",
        target_audience="Jóvenes profesionales, estudiantes y freelancers",
        dimensions={
            "width": 8.0,
            "length": 12.0,
            "height": 3.5
        },
        additional_info="Quiero un ambiente acogedor para trabajar y socializar",
        show_timing=True,
        show_stats=True,
        export=True
    )
    
    if design1 is None:
        logger.warning("No se pudo generar el diseño")
        _print_message("\n⚠️  No se pudo generar el diseño. Revise los errores arriba.", "warning")
        return
    
    _display_design_metadata(design1)
    
    # Opcional: Generar segundo diseño para comparar
    if os.getenv("PSD_GENERATE_SECOND_DESIGN", "false").lower() == "true":
        logger.info("Generando segundo diseño para comparación")
        design2 = await generate_store_design_example(
            store_name="Café Clásico Elegante",
            store_type=StoreType.CAFE,
            style=DesignStyle.CLASSIC,
            budget_range="alto",
            location="Zona residencial exclusiva",
            target_audience="Profesionales y empresarios",
            dimensions={
                "width": 10.0,
                "length": 15.0,
                "height": 4.0
            },
            show_timing=False,
            show_stats=False,
            export=False
        )
        
        if design2:
            comparison = compare_designs(design1, design2)
            print_section("🔍 Comparación de Diseños", "=")
            for key, value in comparison.items():
                if isinstance(value, dict) and value.get("different"):
                    print_field(key, f"Design1: {value.get('design1')} | Design2: {value.get('design2')}")
    
    logger.info("Ejemplo de uso completado exitosamente")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Proceso interrumpido por el usuario")
        if RICH_AVAILABLE and console:
            console.print("\n\n[yellow]⚠️  Proceso interrumpido por el usuario[/yellow]")
        else:
            print("\n\n⚠️  Proceso interrumpido por el usuario", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        logger.exception("Error fatal en ejecución")
        if RICH_AVAILABLE and console:
            error_panel = Panel(
                f"[bold red]❌ Error fatal:[/bold red] {type(e).__name__}: {e}",
                title="Error Fatal",
                border_style="red",
                box=box.ROUNDED
            )
            console.print(error_panel)
        else:
            print(f"\n❌ Error fatal: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
