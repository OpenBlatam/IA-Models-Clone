#!/usr/bin/env python3
"""
Scraper para extraer todos los elementos de antigravity.google y guardarlos en JSON
"""

import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

def scrape_antigravity():
    """Extrae todos los elementos de antigravity.google"""
    
    # Configurar Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = None
    try:
        # Inicializar driver
        driver = webdriver.Chrome(options=chrome_options)
        driver.get('https://antigravity.google/')
        
        # Esperar a que la página cargue
        time.sleep(5)
        
        # Estructura para almacenar todos los elementos
        data = {
            'url': 'https://antigravity.google/',
            'title': driver.title,
            'header': {},
            'hero_section': {},
            'features_section': {},
            'use_cases_section': {},
            'pricing_section': {},
            'blog_section': {},
            'download_section': {},
            'footer': {},
            'styles': {},
            'animations': {}
        }
        
        # Extraer Header
        try:
            header = driver.find_element(By.TAG_NAME, 'header')
            header_data = {
                'logo': {
                    'text': header.find_element(By.TAG_NAME, 'a').text if header.find_elements(By.TAG_NAME, 'a') else '',
                    'href': header.find_element(By.TAG_NAME, 'a').get_attribute('href') if header.find_elements(By.TAG_NAME, 'a') else ''
                },
                'navigation': []
            }
            
            # Extraer elementos de navegación
            nav_items = header.find_elements(By.CSS_SELECTOR, 'nav button, nav a')
            for item in nav_items:
                header_data['navigation'].append({
                    'text': item.text,
                    'href': item.get_attribute('href') if item.tag_name == 'a' else None,
                    'type': item.tag_name
                })
            
            data['header'] = header_data
        except Exception as e:
            print(f"Error extrayendo header: {e}")
        
        # Extraer Hero Section
        try:
            hero = driver.find_element(By.CSS_SELECTOR, 'section:first-of-type, [class*="hero"]')
            hero_data = {
                'logo': {
                    'text': '',
                    'icon': ''
                },
                'headline': '',
                'subheadline': '',
                'buttons': []
            }
            
            # Buscar logo
            logo_elements = hero.find_elements(By.CSS_SELECTOR, 'svg, [class*="logo"]')
            if logo_elements:
                hero_data['logo']['icon'] = 'present'
            
            # Buscar texto del logo
            logo_text = hero.find_elements(By.CSS_SELECTOR, 'span, h1, [class*="logo"]')
            if logo_text:
                hero_data['logo']['text'] = logo_text[0].text if logo_text else ''
            
            # Buscar headline
            headlines = hero.find_elements(By.CSS_SELECTOR, 'h1')
            if headlines:
                hero_data['headline'] = headlines[0].text
            
            # Buscar subheadline
            subheadlines = hero.find_elements(By.CSS_SELECTOR, 'p, [class*="subtitle"]')
            if subheadlines:
                hero_data['subheadline'] = subheadlines[0].text
            
            # Buscar botones
            buttons = hero.find_elements(By.CSS_SELECTOR, 'button, a[class*="button"]')
            for btn in buttons:
                hero_data['buttons'].append({
                    'text': btn.text,
                    'href': btn.get_attribute('href'),
                    'class': btn.get_attribute('class')
                })
            
            data['hero_section'] = hero_data
        except Exception as e:
            print(f"Error extrayendo hero section: {e}")
        
        # Extraer todas las secciones
        sections = driver.find_elements(By.CSS_SELECTOR, 'section')
        for i, section in enumerate(sections):
            try:
                section_data = {
                    'heading': '',
                    'content': [],
                    'links': [],
                    'buttons': []
                }
                
                # Buscar headings
                headings = section.find_elements(By.CSS_SELECTOR, 'h1, h2, h3, h4, h5, h6')
                if headings:
                    section_data['heading'] = headings[0].text
                
                # Buscar párrafos
                paragraphs = section.find_elements(By.CSS_SELECTOR, 'p, li')
                for p in paragraphs:
                    if p.text.strip():
                        section_data['content'].append(p.text.strip())
                
                # Buscar enlaces
                links = section.find_elements(By.CSS_SELECTOR, 'a')
                for link in links:
                    if link.text.strip():
                        section_data['links'].append({
                            'text': link.text.strip(),
                            'href': link.get_attribute('href')
                        })
                
                # Buscar botones
                buttons = section.find_elements(By.CSS_SELECTOR, 'button')
                for btn in buttons:
                    if btn.text.strip():
                        section_data['buttons'].append({
                            'text': btn.text.strip(),
                            'class': btn.get_attribute('class')
                        })
                
                # Asignar a la sección correspondiente según el índice
                if i == 0:
                    data['hero_section'].update(section_data)
                elif i == 1:
                    data['features_section'] = section_data
                elif i == 2:
                    data['use_cases_section'] = section_data
                elif i == 3:
                    data['pricing_section'] = section_data
                elif i == 4:
                    data['blog_section'] = section_data
                elif i == 5:
                    data['download_section'] = section_data
                    
            except Exception as e:
                print(f"Error extrayendo sección {i}: {e}")
        
        # Extraer Footer
        try:
            footer = driver.find_element(By.TAG_NAME, 'footer')
            footer_data = {
                'links': [],
                'text': footer.text
            }
            
            links = footer.find_elements(By.CSS_SELECTOR, 'a')
            for link in links:
                footer_data['links'].append({
                    'text': link.text.strip(),
                    'href': link.get_attribute('href')
                })
            
            data['footer'] = footer_data
        except Exception as e:
            print(f"Error extrayendo footer: {e}")
        
        # Extraer estilos CSS
        try:
            styles = driver.execute_script("""
                var styles = {};
                var sheets = document.styleSheets;
                for (var i = 0; i < sheets.length; i++) {
                    try {
                        var rules = sheets[i].cssRules || sheets[i].rules;
                        for (var j = 0; j < rules.length; j++) {
                            var rule = rules[j];
                            if (rule.selectorText) {
                                styles[rule.selectorText] = rule.style.cssText;
                            }
                        }
                    } catch(e) {
                        console.log('Error accessing stylesheet:', e);
                    }
                }
                return styles;
            """)
            data['styles'] = styles
        except Exception as e:
            print(f"Error extrayendo estilos: {e}")
        
        # Extraer información de animaciones
        try:
            animations = driver.execute_script("""
                var animations = {};
                var sheets = document.styleSheets;
                for (var i = 0; i < sheets.length; i++) {
                    try {
                        var rules = sheets[i].cssRules || sheets[i].rules;
                        for (var j = 0; j < rules.length; j++) {
                            var rule = rules[j];
                            if (rule.type === 7 && rule.name) { // KEYFRAMES_RULE
                                animations[rule.name] = {};
                                for (var k = 0; k < rule.cssRules.length; k++) {
                                    var keyframe = rule.cssRules[k];
                                    animations[rule.name][keyframe.keyText] = keyframe.style.cssText;
                                }
                            }
                        }
                    } catch(e) {
                        console.log('Error accessing animations:', e);
                    }
                }
                return animations;
            """)
            data['animations'] = animations
        except Exception as e:
            print(f"Error extrayendo animaciones: {e}")
        
        # Extraer todos los elementos con sus atributos
        try:
            all_elements = driver.execute_script("""
                var elements = [];
                var all = document.querySelectorAll('*');
                for (var i = 0; i < all.length; i++) {
                    var el = all[i];
                    if (el.tagName && el.tagName !== 'SCRIPT' && el.tagName !== 'STYLE') {
                        var attrs = {};
                        for (var j = 0; j < el.attributes.length; j++) {
                            attrs[el.attributes[j].name] = el.attributes[j].value;
                        }
                        elements.push({
                            tag: el.tagName,
                            text: el.textContent ? el.textContent.trim().substring(0, 100) : '',
                            attributes: attrs,
                            className: el.className,
                            id: el.id
                        });
                    }
                }
                return elements;
            """)
            data['all_elements'] = all_elements
        except Exception as e:
            print(f"Error extrayendo todos los elementos: {e}")
        
        return data
        
    except Exception as e:
        print(f"Error general: {e}")
        return None
        
    finally:
        if driver:
            driver.quit()

if __name__ == '__main__':
    print("Iniciando scraper de antigravity.google...")
    data = scrape_antigravity()
    
    if data:
        # Guardar en JSON
        output_file = 'antigravity_elements.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Datos extraídos y guardados en {output_file}")
        print(f"📊 Total de elementos extraídos: {len(data.get('all_elements', []))}")
    else:
        print("❌ Error al extraer datos")



