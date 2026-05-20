#!/usr/bin/env python3
"""
Genera un reporte detallado en texto legible de todos los elementos
"""

import json

def generate_report():
    """Genera un reporte detallado en texto"""
    
    try:
        with open('antigravity_analysis_detailed.json', 'r', encoding='utf-8') as f:
            analysis = json.load(f)
        
        with open('antigravity_elements_complete.json', 'r', encoding='utf-8') as f:
            elements = json.load(f)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("REPORTE DETALLADO - ANTIGRAVITY.GOOGLE")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Resumen general
        report_lines.append("## RESUMEN GENERAL")
        report_lines.append("-" * 80)
        report_lines.append(f"URL: {analysis['summary']['url']}")
        report_lines.append(f"Titulo: {analysis['summary']['title']}")
        report_lines.append(f"Total de elementos extraidos: {analysis['summary']['total_elements']}")
        report_lines.append("")
        
        # Roles encontrados
        report_lines.append("## ROLES ENCONTRADOS")
        report_lines.append("-" * 80)
        for role, count in sorted(analysis['summary']['roles_found'].items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"  {role:20s}: {count:3d} elementos")
        report_lines.append("")
        
        # Estadísticas
        report_lines.append("## ESTADISTICAS DETALLADAS")
        report_lines.append("-" * 80)
        stats = analysis['statistics']
        report_lines.append(f"Total de botones: {stats['total_buttons']}")
        report_lines.append(f"Total de enlaces: {stats['total_links']}")
        report_lines.append(f"Total de secciones: {stats['total_sections']}")
        report_lines.append(f"Total de items de navegacion: {stats['total_navigation_items']}")
        report_lines.append(f"Total de items de menu: {stats['total_menu_items']}")
        report_lines.append(f"Total de items de lista: {stats['total_list_items']}")
        report_lines.append("")
        report_lines.append(f"Profundidad maxima del DOM: {stats['path_analysis']['max_depth']}")
        report_lines.append(f"Profundidad minima del DOM: {stats['path_analysis']['min_depth']}")
        report_lines.append(f"Profundidad promedio: {stats['path_analysis']['avg_depth']:.2f}")
        report_lines.append("")
        
        # Botones
        report_lines.append("## BOTONES ENCONTRADOS")
        report_lines.append("-" * 80)
        for i, btn in enumerate(analysis['buttons'], 1):
            report_lines.append(f"{i:2d}. {btn['name']}")
            report_lines.append(f"    Ref: {btn['ref']}")
            report_lines.append(f"    Path: {btn['path']}")
            report_lines.append("")
        
        # Enlaces
        report_lines.append("## ENLACES ENCONTRADOS")
        report_lines.append("-" * 80)
        for i, link in enumerate(analysis['links'], 1):
            report_lines.append(f"{i:2d}. {link['name']}")
            report_lines.append(f"    Ref: {link['ref']}")
            report_lines.append(f"    Path: {link['path']}")
            report_lines.append("")
        
        # Secciones
        report_lines.append("## SECCIONES ENCONTRADAS")
        report_lines.append("-" * 80)
        for i, sec in enumerate(analysis['sections'], 1):
            report_lines.append(f"{i}. {sec['name'] or 'Sin nombre'}")
            report_lines.append(f"   Ref: {sec['ref']}")
            report_lines.append(f"   Path: {sec['path']}")
            report_lines.append("")
        
        # Header
        report_lines.append("## DETALLES DEL HEADER")
        report_lines.append("-" * 80)
        header = analysis['header_details']
        report_lines.append(f"Ref: {header.get('ref', 'N/A')}")
        report_lines.append(f"Path: {header.get('path', 'N/A')}")
        report_lines.append(f"Numero de hijos: {header.get('children_count', 0)}")
        report_lines.append(f"Botones en header: {len(header.get('buttons_in_header', []))}")
        report_lines.append(f"Enlaces en header: {len(header.get('links_in_header', []))}")
        report_lines.append("")
        
        # Hero Section
        report_lines.append("## DETALLES DEL HERO SECTION")
        report_lines.append("-" * 80)
        hero = analysis['hero_section_details']
        report_lines.append(f"Ref: {hero.get('ref', 'N/A')}")
        report_lines.append(f"Path: {hero.get('path', 'N/A')}")
        report_lines.append(f"Numero de hijos: {hero.get('children_count', 0)}")
        report_lines.append(f"Botones en hero: {len(hero.get('buttons_in_hero', []))}")
        report_lines.append(f"Enlaces en hero: {len(hero.get('links_in_hero', []))}")
        report_lines.append("")
        
        # Footer
        report_lines.append("## DETALLES DEL FOOTER")
        report_lines.append("-" * 80)
        footer = analysis['footer_details']
        if footer:
            report_lines.append(f"Ref: {footer.get('ref', 'N/A')}")
            report_lines.append(f"Path: {footer.get('path', 'N/A')}")
            report_lines.append(f"Numero de hijos: {footer.get('children_count', 0)}")
            report_lines.append(f"Enlaces en footer: {len(footer.get('links_in_footer', []))}")
        else:
            report_lines.append("Footer no encontrado en el snapshot")
        report_lines.append("")
        
        # Botones únicos
        report_lines.append("## BOTONES UNICOS (SIN DUPLICADOS)")
        report_lines.append("-" * 80)
        for btn_name in sorted(stats['unique_button_names']):
            if btn_name:
                report_lines.append(f"  - {btn_name}")
        report_lines.append("")
        
        # Enlaces únicos
        report_lines.append("## ENLACES UNICOS (SIN DUPLICADOS)")
        report_lines.append("-" * 80)
        for link_name in sorted(stats['unique_link_names']):
            if link_name:
                report_lines.append(f"  - {link_name}")
        report_lines.append("")
        
        # Estructura del DOM
        report_lines.append("## ESTRUCTURA DEL DOM")
        report_lines.append("-" * 80)
        report_lines.append("Ejemplos de paths encontrados:")
        all_elements = elements.get('all_elements', [])
        sample_paths = [e.get('path', '') for e in all_elements[:10]]
        for path in sample_paths:
            report_lines.append(f"  {path}")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("FIN DEL REPORTE")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
        
    except Exception as e:
        print(f"[ERROR] Error generando reporte: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    print("Generando reporte detallado...")
    report = generate_report()
    
    if report:
        output_file = 'antigravity_detailed_report.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[OK] Reporte detallado guardado en {output_file}")
        print("\n" + "=" * 80)
        print(report)
    else:
        print("[ERROR] Error al generar reporte")



