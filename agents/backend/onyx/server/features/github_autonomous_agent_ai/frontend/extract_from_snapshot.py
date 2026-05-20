#!/usr/bin/env python3
"""
Extrae todos los elementos del snapshot del navegador y los guarda en JSON
"""

import json
import yaml
import os

def extract_from_snapshot():
    """Extrae elementos del snapshot del navegador"""
    
    # Buscar el archivo de snapshot más reciente
    snapshot_dir = os.path.expanduser("~/.cursor/browser-logs")
    snapshots = []
    
    if os.path.exists(snapshot_dir):
        for file in os.listdir(snapshot_dir):
            if file.startswith("snapshot-") and file.endswith(".log"):
                snapshots.append(os.path.join(snapshot_dir, file))
        
        snapshots.sort(reverse=True)  # Más reciente primero
    
    if not snapshots:
        print("[ERROR] No se encontraron snapshots del navegador")
        return None
    
    # Leer el snapshot más reciente
    latest_snapshot = snapshots[0]
    print(f"[INFO] Leyendo snapshot: {latest_snapshot}")
    
    try:
        with open(latest_snapshot, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parsear YAML
        data = yaml.safe_load(content)
        
        # Estructura para almacenar elementos extraídos
        extracted = {
            'url': 'https://antigravity.google/',
            'title': 'Google Antigravity',
            'header': {},
            'hero_section': {},
            'sections': [],
            'footer': {},
            'all_elements': []
        }
        
        def extract_elements(node, path=""):
            """Recursivamente extrae elementos del árbol"""
            elements = []
            
            if isinstance(node, dict):
                role = node.get('role', '')
                name = node.get('name', '')
                ref = node.get('ref', '')
                children = node.get('children', [])
                
                element = {
                    'role': role,
                    'name': name,
                    'ref': ref,
                    'path': path
                }
                
                elements.append(element)
                
                # Procesar hijos
                for i, child in enumerate(children):
                    child_path = f"{path}/{role}[{i}]" if path else f"{role}[{i}]"
                    elements.extend(extract_elements(child, child_path))
            
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    elements.extend(extract_elements(item, f"{path}[{i}]"))
            
            return elements
        
        # Extraer todos los elementos
        all_elements = extract_elements(data)
        extracted['all_elements'] = all_elements
        extracted['total_elements'] = len(all_elements)
        
        # Categorizar elementos
        for elem in all_elements:
            role = elem.get('role', '')
            name = elem.get('name', '')
            
            if role == 'banner' or 'header' in name.lower():
                extracted['header'] = elem
            elif role == 'main':
                extracted['hero_section'] = elem
            elif 'footer' in role.lower() or 'footer' in name.lower():
                extracted['footer'] = elem
        
        # Agrupar por roles
        roles = {}
        for elem in all_elements:
            role = elem.get('role', 'unknown')
            if role not in roles:
                roles[role] = []
            roles[role].append(elem)
        
        extracted['by_role'] = {role: len(elems) for role, elems in roles.items()}
        extracted['roles_detail'] = roles
        
        return extracted
        
    except Exception as e:
        print(f"[ERROR] Error procesando snapshot: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    print("Extrayendo elementos del snapshot del navegador...")
    data = extract_from_snapshot()
    
    if data:
        # Guardar en JSON
        output_file = 'antigravity_elements_complete.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Datos extraidos y guardados en {output_file}")
        print(f"[INFO] Total de elementos: {data.get('total_elements', 0)}")
        print(f"[INFO] Roles encontrados: {len(data.get('by_role', {}))}")
        for role, count in data.get('by_role', {}).items():
            print(f"  - {role}: {count}")
    else:
        print("[ERROR] Error al extraer datos")



