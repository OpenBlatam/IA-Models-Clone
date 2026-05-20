#!/usr/bin/env python3
"""
Scraper usando solo biblioteca estándar de Python para extraer elementos de antigravity.google
"""

import json
import urllib.request
import urllib.parse
import re
import gzip
from html.parser import HTMLParser

class ElementParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []
        self.current_tag = None
        self.current_attrs = {}
        self.current_text = ""
        self.stack = []
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.current_attrs = dict(attrs)
        self.current_text = ""
        self.stack.append({
            'tag': tag,
            'attributes': self.current_attrs,
            'text': '',
            'children': []
        })
        
    def handle_endtag(self, tag):
        if self.stack:
            element = self.stack.pop()
            if element['text'].strip():
                element['text'] = element['text'].strip()
            if self.stack:
                self.stack[-1]['children'].append(element)
            else:
                self.elements.append(element)
                
    def handle_data(self, data):
        if self.stack:
            self.stack[-1]['text'] += data

def scrape_antigravity():
    """Extrae todos los elementos de antigravity.google"""
    
    url = 'https://antigravity.google/'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()
            # Intentar descomprimir gzip
            try:
                html = gzip.decompress(content).decode('utf-8')
            except:
                html = content.decode('utf-8')
        
        # Parsear HTML
        parser = ElementParser()
        parser.feed(html)
        
        data = {
            'url': url,
            'title': re.search(r'<title>(.*?)</title>', html, re.DOTALL),
            'header': {},
            'hero_section': {},
            'sections': [],
            'footer': {},
            'all_elements': []
        }
        
        # Extraer título
        title_match = re.search(r'<title>(.*?)</title>', html, re.DOTALL)
        if title_match:
            data['title'] = title_match.group(1).strip()
        
        # Extraer header
        header_match = re.search(r'<header[^>]*>(.*?)</header>', html, re.DOTALL)
        if header_match:
            header_html = header_match.group(1)
            data['header'] = {
                'html': header_html[:500],
                'links': re.findall(r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>', header_html, re.DOTALL),
                'buttons': re.findall(r'<button[^>]*>(.*?)</button>', header_html, re.DOTALL)
            }
        
        # Extraer todas las secciones
        sections = re.findall(r'<section[^>]*>(.*?)</section>', html, re.DOTALL)
        for i, section_html in enumerate(sections):
            section_data = {
                'index': i,
                'headings': re.findall(r'<h[1-6][^>]*>(.*?)</h[1-6]>', section_html, re.DOTALL),
                'paragraphs': re.findall(r'<p[^>]*>(.*?)</p>', section_html, re.DOTALL),
                'links': re.findall(r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>', section_html, re.DOTALL),
                'buttons': re.findall(r'<button[^>]*>(.*?)</button>', section_html, re.DOTALL),
                'lists': re.findall(r'<li[^>]*>(.*?)</li>', section_html, re.DOTALL)
            }
            
            # Limpiar texto
            section_data['headings'] = [re.sub(r'<[^>]+>', '', h).strip() for h in section_data['headings']]
            section_data['paragraphs'] = [re.sub(r'<[^>]+>', '', p).strip() for p in section_data['paragraphs']]
            section_data['buttons'] = [re.sub(r'<[^>]+>', '', b).strip() for b in section_data['buttons']]
            section_data['lists'] = [re.sub(r'<[^>]+>', '', l).strip() for l in section_data['lists']]
            
            data['sections'].append(section_data)
            
            # Asignar a secciones específicas
            if i == 0:
                data['hero_section'] = section_data
        
        # Extraer footer
        footer_match = re.search(r'<footer[^>]*>(.*?)</footer>', html, re.DOTALL)
        if footer_match:
            footer_html = footer_match.group(1)
            data['footer'] = {
                'links': re.findall(r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>', footer_html, re.DOTALL),
                'text': re.sub(r'<[^>]+>', '', footer_html).strip()
            }
        
        # Extraer todos los elementos con regex
        all_tags = re.findall(r'<(\w+)([^>]*)>(.*?)</\1>', html, re.DOTALL)
        for tag, attrs, content in all_tags[:1000]:  # Limitar a 1000 elementos
            if tag not in ['script', 'style']:
                attrs_dict = {}
                attr_matches = re.findall(r'(\w+)=["\']([^"\']*)["\']', attrs)
                for attr_name, attr_value in attr_matches:
                    attrs_dict[attr_name] = attr_value
                
                text = re.sub(r'<[^>]+>', '', content).strip()[:200]
                
                data['all_elements'].append({
                    'tag': tag,
                    'attributes': attrs_dict,
                    'text': text,
                    'class': attrs_dict.get('class', ''),
                    'id': attrs_dict.get('id', '')
                })
        
        data['total_elements'] = len(data['all_elements'])
        
        return data
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    print("Iniciando scraper de antigravity.google...")
    data = scrape_antigravity()
    
    if data:
        # Guardar en JSON
        output_file = 'antigravity_elements.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Datos extraidos y guardados en {output_file}")
        print(f"[INFO] Total de elementos: {data.get('total_elements', 0)}")
        print(f"[INFO] Total de secciones: {len(data.get('sections', []))}")
        if data.get('sections'):
            print(f"[INFO] Total de enlaces: {sum(len(s.get('links', [])) for s in data.get('sections', []))}")
            print(f"[INFO] Total de botones: {sum(len(s.get('buttons', [])) for s in data.get('sections', []))}")
    else:
        print("[ERROR] Error al extraer datos")

