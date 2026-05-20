#!/usr/bin/env python3
"""
Scraper simple usando requests y BeautifulSoup para extraer elementos de antigravity.google
"""

import json
import requests
from bs4 import BeautifulSoup
import re

def scrape_antigravity_simple():
    """Extrae todos los elementos de antigravity.google usando requests"""
    
    url = 'https://antigravity.google/'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        data = {
            'url': url,
            'title': soup.title.string if soup.title else '',
            'header': {},
            'hero_section': {},
            'sections': [],
            'footer': {},
            'styles': {},
            'scripts': []
        }
        
        # Extraer Header
        header = soup.find('header')
        if header:
            header_data = {
                'logo': {},
                'navigation': []
            }
            
            # Logo
            logo_link = header.find('a')
            if logo_link:
                header_data['logo'] = {
                    'text': logo_link.get_text(strip=True),
                    'href': logo_link.get('href', '')
                }
            
            # Navegación
            nav = header.find('nav')
            if nav:
                nav_items = nav.find_all(['a', 'button'])
                for item in nav_items:
                    header_data['navigation'].append({
                        'text': item.get_text(strip=True),
                        'href': item.get('href'),
                        'tag': item.name,
                        'class': item.get('class', [])
                    })
            
            data['header'] = header_data
        
        # Extraer Hero Section
        sections = soup.find_all('section')
        for i, section in enumerate(sections):
            section_data = {
                'index': i,
                'heading': '',
                'headings': [],
                'paragraphs': [],
                'links': [],
                'buttons': [],
                'images': [],
                'class': section.get('class', []),
                'id': section.get('id', '')
            }
            
            # Headings
            headings = section.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            for h in headings:
                heading_text = h.get_text(strip=True)
                section_data['headings'].append({
                    'tag': h.name,
                    'text': heading_text,
                    'class': h.get('class', [])
                })
                if not section_data['heading']:
                    section_data['heading'] = heading_text
            
            # Párrafos
            paragraphs = section.find_all('p')
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text:
                    section_data['paragraphs'].append({
                        'text': text,
                        'class': p.get('class', [])
                    })
            
            # Listas
            lists = section.find_all(['ul', 'ol'])
            for ul in lists:
                items = ul.find_all('li')
                for li in items:
                    text = li.get_text(strip=True)
                    if text:
                        section_data['paragraphs'].append({
                            'text': text,
                            'type': 'list-item',
                            'class': li.get('class', [])
                        })
            
            # Enlaces
            links = section.find_all('a')
            for link in links:
                text = link.get_text(strip=True)
                if text:
                    section_data['links'].append({
                        'text': text,
                        'href': link.get('href', ''),
                        'class': link.get('class', [])
                    })
            
            # Botones
            buttons = section.find_all('button')
            for btn in buttons:
                text = btn.get_text(strip=True)
                if text:
                    section_data['buttons'].append({
                        'text': text,
                        'class': btn.get('class', []),
                        'type': btn.get('type', '')
                    })
            
            # Imágenes
            images = section.find_all('img')
            for img in images:
                section_data['images'].append({
                    'src': img.get('src', ''),
                    'alt': img.get('alt', ''),
                    'class': img.get('class', [])
                })
            
            # SVG
            svgs = section.find_all('svg')
            for svg in svgs:
                section_data['images'].append({
                    'type': 'svg',
                    'viewBox': svg.get('viewBox', ''),
                    'class': svg.get('class', [])
                })
            
            data['sections'].append(section_data)
            
            # Asignar a secciones específicas
            if i == 0:
                data['hero_section'] = section_data
            elif 'feature' in str(section.get('class', [])).lower() or i == 1:
                data['features_section'] = section_data
            elif 'use-case' in str(section.get('class', [])).lower() or i == 2:
                data['use_cases_section'] = section_data
            elif 'pricing' in str(section.get('class', [])).lower() or i == 3:
                data['pricing_section'] = section_data
            elif 'blog' in str(section.get('class', [])).lower() or i == 4:
                data['blog_section'] = section_data
            elif 'download' in str(section.get('class', [])).lower() or i == 5:
                data['download_section'] = section_data
        
        # Extraer Footer
        footer = soup.find('footer')
        if footer:
            footer_data = {
                'links': [],
                'text': footer.get_text(strip=True)
            }
            
            links = footer.find_all('a')
            for link in links:
                footer_data['links'].append({
                    'text': link.get_text(strip=True),
                    'href': link.get('href', ''),
                    'class': link.get('class', [])
                })
            
            data['footer'] = footer_data
        
        # Extraer estilos inline
        styles = soup.find_all('style')
        for style in styles:
            data['styles'][f'style_{len(data["styles"])}'] = style.string
        
        # Extraer scripts
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string:
                data['scripts'].append({
                    'src': script.get('src', ''),
                    'content_length': len(script.string),
                    'type': script.get('type', '')
                })
        
        # Extraer todos los elementos con sus atributos
        all_elements = []
        for element in soup.find_all(True):
            if element.name not in ['script', 'style']:
                elem_data = {
                    'tag': element.name,
                    'text': element.get_text(strip=True)[:200] if element.get_text(strip=True) else '',
                    'attributes': dict(element.attrs),
                    'class': element.get('class', []),
                    'id': element.get('id', '')
                }
                all_elements.append(elem_data)
        
        data['all_elements'] = all_elements
        data['total_elements'] = len(all_elements)
        
        return data
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == '__main__':
    print("Iniciando scraper simple de antigravity.google...")
    data = scrape_antigravity_simple()
    
    if data:
        # Guardar en JSON
        output_file = 'antigravity_elements.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Datos extraídos y guardados en {output_file}")
        print(f"📊 Total de elementos: {data.get('total_elements', 0)}")
        print(f"📄 Total de secciones: {len(data.get('sections', []))}")
        print(f"🔗 Total de enlaces: {sum(len(s.get('links', [])) for s in data.get('sections', []))}")
        print(f"🔘 Total de botones: {sum(len(s.get('buttons', [])) for s in data.get('sections', []))}")
    else:
        print("❌ Error al extraer datos")



