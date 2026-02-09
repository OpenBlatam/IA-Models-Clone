#!/usr/bin/env python3
"""
Analiza el JSON de elementos extraídos y genera un reporte detallado
"""

import json
from collections import Counter

def analyze_elements():
    """Analiza los elementos extraídos y genera un reporte detallado"""
    
    try:
        with open('antigravity_elements_complete.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        report = {
            'summary': {},
            'header_details': {},
            'hero_section_details': {},
            'navigation_items': [],
            'buttons': [],
            'links': [],
            'sections': [],
            'footer_details': {},
            'statistics': {}
        }
        
        # Resumen general
        report['summary'] = {
            'total_elements': data.get('total_elements', 0),
            'url': data.get('url', ''),
            'title': data.get('title', ''),
            'roles_found': data.get('by_role', {})
        }
        
        # Analizar elementos por rol
        all_elements = data.get('all_elements', [])
        
        # Extraer botones
        buttons = [e for e in all_elements if e.get('role') == 'button']
        for btn in buttons:
            report['buttons'].append({
                'name': btn.get('name', ''),
                'ref': btn.get('ref', ''),
                'path': btn.get('path', '')
            })
        
        # Extraer enlaces
        links = [e for e in all_elements if e.get('role') == 'link']
        for link in links:
            report['links'].append({
                'name': link.get('name', ''),
                'ref': link.get('ref', ''),
                'path': link.get('path', '')
            })
        
        # Extraer secciones
        sections = [e for e in all_elements if e.get('role') == 'section']
        for sec in sections:
            report['sections'].append({
                'name': sec.get('name', ''),
                'ref': sec.get('ref', ''),
                'path': sec.get('path', '')
            })
        
        # Analizar roles detallados
        roles_detail = data.get('roles_detail', {})
        
        # Detalles del header
        header_elem = data.get('header', {})
        if header_elem:
            header_children = [e for e in all_elements if 'banner' in e.get('path', '')]
            report['header_details'] = {
                'ref': header_elem.get('ref', ''),
                'path': header_elem.get('path', ''),
                'children_count': len(header_children),
                'buttons_in_header': [b for b in buttons if 'banner' in b.get('path', '')],
                'links_in_header': [l for l in links if 'banner' in l.get('path', '')]
            }
        
        # Detalles del hero section
        hero_elem = data.get('hero_section', {})
        if hero_elem:
            hero_children = [e for e in all_elements if hero_elem.get('ref') in str(e.get('path', ''))]
            report['hero_section_details'] = {
                'ref': hero_elem.get('ref', ''),
                'path': hero_elem.get('path', ''),
                'children_count': len(hero_children),
                'buttons_in_hero': [b for b in buttons if hero_elem.get('ref') in str(b.get('path', ''))],
                'links_in_hero': [l for l in links if hero_elem.get('ref') in str(l.get('path', ''))]
            }
        
        # Detalles del footer
        footer_elems = [e for e in all_elements if e.get('role') == 'contentinfo']
        if footer_elems:
            footer_elem = footer_elems[0]
            footer_children = [e for e in all_elements if footer_elem.get('ref') in str(e.get('path', ''))]
            report['footer_details'] = {
                'ref': footer_elem.get('ref', ''),
                'path': footer_elem.get('path', ''),
                'children_count': len(footer_children),
                'links_in_footer': [l for l in links if footer_elem.get('ref') in str(l.get('path', ''))]
            }
        
        # Estadísticas
        report['statistics'] = {
            'total_buttons': len(buttons),
            'total_links': len(links),
            'total_sections': len(sections),
            'total_navigation_items': len([e for e in all_elements if e.get('role') == 'navigation']),
            'total_menu_items': len([e for e in all_elements if e.get('role') == 'menuitem']),
            'total_list_items': len([e for e in all_elements if e.get('role') == 'listitem']),
            'button_names': [b.get('name', '') for b in buttons if b.get('name')],
            'link_names': [l.get('name', '') for l in links if l.get('name')],
            'unique_button_names': list(set([b.get('name', '') for b in buttons if b.get('name')])),
            'unique_link_names': list(set([l.get('name', '') for l in links if l.get('name')]))
        }
        
        # Análisis de paths
        paths = [e.get('path', '') for e in all_elements]
        path_depths = [len(p.split('/')) for p in paths]
        report['statistics']['path_analysis'] = {
            'max_depth': max(path_depths) if path_depths else 0,
            'min_depth': min(path_depths) if path_depths else 0,
            'avg_depth': sum(path_depths) / len(path_depths) if path_depths else 0
        }
        
        return report
        
    except Exception as e:
        print(f"[ERROR] Error analizando elementos: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    print("Analizando elementos extraidos...")
    report = analyze_elements()
    
    if report:
        # Guardar reporte detallado
        output_file = 'antigravity_analysis_detailed.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Analisis detallado guardado en {output_file}")
        print("\n=== RESUMEN ===")
        print(f"Total de elementos: {report['summary']['total_elements']}")
        print(f"URL: {report['summary']['url']}")
        print(f"Titulo: {report['summary']['title']}")
        print(f"\n=== ESTADISTICAS ===")
        print(f"Total de botones: {report['statistics']['total_buttons']}")
        print(f"Total de enlaces: {report['statistics']['total_links']}")
        print(f"Total de secciones: {report['statistics']['total_sections']}")
        print(f"Total de items de navegacion: {report['statistics']['total_navigation_items']}")
        print(f"Total de items de menu: {report['statistics']['total_menu_items']}")
        print(f"Total de items de lista: {report['statistics']['total_list_items']}")
        print(f"\n=== BOTONES UNICOS ===")
        for btn_name in report['statistics']['unique_button_names']:
            print(f"  - {btn_name}")
        print(f"\n=== ENLACES UNICOS ===")
        for link_name in report['statistics']['unique_link_names'][:20]:  # Primeros 20
            print(f"  - {link_name}")
        print(f"\n=== ESTRUCTURA ===")
        print(f"Profundidad maxima del DOM: {report['statistics']['path_analysis']['max_depth']}")
        print(f"Profundidad promedio: {report['statistics']['path_analysis']['avg_depth']:.2f}")
    else:
        print("[ERROR] Error al analizar elementos")



