# Mejoras de Análisis Avanzado Finales - Document Analyzer

## Resumen

Se han agregado tres sistemas de análisis avanzado finales para completar el ecosistema del Document Analyzer:

1. **Sistema de Detección de Plagio**
2. **Análisis de Estructura Avanzado**
3. **Optimización Automática de Documentos**

---

## 1. Sistema de Detección de Plagio

### Características

- **Detección de similitud**: Identifica contenido similar entre documentos
- **Fingerprinting avanzado**: Usa n-gramas para detectar coincidencias
- **Múltiples niveles de coincidencia**: Exacto, cercano, parafraseado
- **Reportes detallados**: Incluye posiciones, porcentajes y niveles de riesgo
- **Base de documentos de referencia**: Compara contra documentos conocidos

### Uso

```python
from analizador_de_documentos.core.document_analyzer import DocumentAnalyzer

analyzer = DocumentAnalyzer()

# Agregar documentos de referencia
analyzer.add_reference_document("ref_1", "Contenido del documento de referencia...")
analyzer.add_reference_document("ref_2", "Otro documento de referencia...")

# Detectar plagio
plagiarism_report = await analyzer.detect_plagiarism(
    document_id="doc_123",
    content="Contenido a verificar...",
    threshold=0.7,  # Umbral de similitud
    check_references=True
)

print(f"Similitud general: {plagiarism_report.overall_similarity:.2%}")
print(f"Porcentaje de plagio: {plagiarism_report.plagiarism_percentage:.2f}%")
print(f"Nivel de riesgo: {plagiarism_report.risk_level}")

for match in plagiarism_report.matches:
    print(f"\nCoincidencia con {match.source_document_id}:")
    print(f"  Tipo: {match.match_type}")
    print(f"  Score: {match.similarity_score:.2%}")
    print(f"  Texto original: {match.source_content[:100]}...")
    print(f"  Texto coincidente: {match.matched_content[:100]}...")
```

### Niveles de Riesgo

- **critical**: Similitud >= 90% o porcentaje >= 50%
- **high**: Similitud >= 70% o porcentaje >= 30%
- **medium**: Similitud >= 50% o porcentaje >= 15%
- **low**: Por debajo de los umbrales anteriores

---

## 2. Análisis de Estructura Avanzado

### Características

- **Extracción de secciones**: Identifica y estructura secciones automáticamente
- **Análisis de jerarquía**: Detecta niveles de encabezados y subsecciones
- **Detección de elementos especiales**: TOC, índice, bibliografía
- **Score de estructura**: Evalúa la calidad de la estructura
- **Soporte múltiples formatos**: Markdown, numérico, mayúsculas

### Uso

```python
# Analizar estructura
structure_analysis = await analyzer.analyze_structure_advanced(
    document_id="doc_123",
    content="""# Título Principal

## Sección 1
Contenido de la sección 1...

### Subsección 1.1
Más contenido...

## Sección 2
Contenido de la sección 2...
"""
)

print(f"Total de secciones: {structure_analysis.total_sections}")
print(f"Profundidad de jerarquía: {structure_analysis.hierarchy_depth}")
print(f"Score de estructura: {structure_analysis.structure_score:.2%}")
print(f"Tiene tabla de contenidos: {structure_analysis.has_table_of_contents}")
print(f"Tiene índice: {structure_analysis.has_index}")
print(f"Tiene bibliografía: {structure_analysis.has_bibliography}")

# Recorrer secciones
def print_sections(sections, indent=0):
    for section in sections:
        print("  " * indent + f"- {section.title} (Nivel {section.level})")
        if section.subsections:
            print_sections(section.subsections, indent + 1)

print_sections(structure_analysis.sections)
```

### Formatos de Encabezados Soportados

- **Markdown**: `# ## ### ####`
- **Numérico**: `1. 1.1 1.1.1`
- **Mayúsculas**: Títulos en mayúsculas

---

## 3. Optimización Automática de Documentos

### Características

- **Optimización de claridad**: Simplifica frases confusas
- **Optimización de brevedad**: Elimina redundancias
- **Optimización de engagement**: Convierte voz pasiva a activa
- **Objetivos personalizables**: Define qué optimizar
- **Sugerencias priorizadas**: Recomendaciones con prioridad

### Uso

```python
# Optimizar documento
optimization_result = await analyzer.optimize_document_auto(
    document_id="doc_123",
    content="Documento que necesita optimización...",
    optimization_goals=["clarity", "brevity", "engagement"]
)

print(f"Score original: {optimization_result.original_score:.2%}")
print(f"Score optimizado: {optimization_result.optimized_score:.2%}")
print(f"Mejora: {optimization_result.improvement:+.2%}")
print(f"Optimizaciones aplicadas: {optimization_result.applied_optimizations}")

# Ver sugerencias
for suggestion in optimization_result.suggestions:
    print(f"\n[{suggestion.priority.upper()}] {suggestion.description}")
    print(f"  Original: {suggestion.original_text}")
    print(f"  Optimizado: {suggestion.optimized_text}")
    print(f"  Mejora estimada: {suggestion.improvement_score:.2%}")

# Obtener contenido optimizado
optimized_content = optimization_result.optimized_content
```

### Objetivos de Optimización

- **clarity**: Mejora la claridad del texto
- **brevity**: Reduce redundancias y verbosidad
- **engagement**: Mejora el engagement usando voz activa

### Ejemplos de Optimizaciones

**Claridad:**
- "muy muy" → "extremadamente"
- "en el caso de que" → "si"
- "debido al hecho de que" → "porque"

**Brevedad:**
- "completamente lleno" → "lleno"
- "finalizar completamente" → "finalizar"
- "repetir de nuevo" → "repetir"

**Engagement:**
- "fue realizado" → "realizamos"
- "es considerado" → "consideramos"
- "fue implementado" → "implementamos"

---

## Integración Completa

Todos los sistemas están integrados en el `DocumentAnalyzer` principal:

```python
analyzer = DocumentAnalyzer()

# Plagio
analyzer.add_reference_document("ref_1", reference_content)
plagiarism = await analyzer.detect_plagiarism("doc_123", content)

# Estructura
structure = await analyzer.analyze_structure_advanced("doc_123", content)

# Optimización
optimization = await analyzer.optimize_document_auto("doc_123", content, ["clarity"])
```

---

## Archivos Creados

1. **`core/document_plagiarism.py`**: Sistema de detección de plagio
2. **`core/document_structure_advanced.py`**: Análisis de estructura avanzado
3. **`core/document_auto_optimizer.py`**: Optimización automática

---

## Beneficios

### Detección de Plagio
- ✅ Protección de contenido original
- ✅ Detección temprana de problemas
- ✅ Reportes detallados

### Análisis de Estructura
- ✅ Mejora la organización
- ✅ Identifica problemas estructurales
- ✅ Evalúa calidad de estructura

### Optimización Automática
- ✅ Mejora continua
- ✅ Sugerencias específicas
- ✅ Aplicación automática

---

## Casos de Uso

### Verificación de Plagio
```python
# En un sistema educativo
plagiarism_report = await analyzer.detect_plagiarism("student_essay", essay_content)
if plagiarism_report.risk_level in ["high", "critical"]:
    send_alert_to_teacher(plagiarism_report)
```

### Análisis de Documentos Técnicos
```python
# Para documentos técnicos
structure = await analyzer.analyze_structure_advanced("tech_doc", content)
if structure.structure_score < 0.7:
    generate_structure_recommendations(structure)
```

### Optimización Automática
```python
# Optimizar antes de publicar
optimization = await analyzer.optimize_document_auto("blog_post", content, ["clarity", "engagement"])
publish_content(optimization.optimized_content)
```

---

## Resumen del Sistema Completo

El Document Analyzer ahora incluye:

- ✅ **44+ módulos principales**
- ✅ **130+ funcionalidades**
- ✅ **Detección de plagio**
- ✅ **Análisis de estructura avanzado**
- ✅ **Optimización automática**
- ✅ **Y todas las funcionalidades anteriores**

**Sistema completo y listo para producción enterprise con capacidades de análisis avanzado.**


