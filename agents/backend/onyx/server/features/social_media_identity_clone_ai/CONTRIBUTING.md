# 🤝 Guía de Contribución - Social Media Identity Clone AI

¡Gracias por tu interés en contribuir! Esta guía te ayudará a empezar.

## Cómo Contribuir

### Reportar Bugs

1. Verifica que el bug no haya sido reportado ya
2. Crea un issue con:
   - Descripción clara del problema
   - Pasos para reproducir
   - Comportamiento esperado vs actual
   - Versión del sistema
   - Logs relevantes

### Sugerir Mejoras

1. Crea un issue con:
   - Descripción de la mejora
   - Casos de uso
   - Beneficios esperados
   - Alternativas consideradas

### Pull Requests

1. **Fork** el repositorio
2. **Crea** una rama (`git checkout -b feature/AmazingFeature`)
3. **Commit** tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. **Push** a la rama (`git push origin feature/AmazingFeature`)
5. **Abre** un Pull Request

## Estándares de Código

### Python

- Seguir PEP 8
- Usar type hints
- Documentar funciones y clases
- Máximo 100 caracteres por línea

### Estructura

```
feature/
├── __init__.py
├── service.py
├── models.py
└── tests/
    └── test_service.py
```

### Tests

- Escribir tests para nuevas funcionalidades
- Mantener cobertura > 80%
- Usar pytest

```python
def test_new_feature():
    """Test de nueva funcionalidad"""
    # Arrange
    service = NewService()
    
    # Act
    result = service.do_something()
    
    # Assert
    assert result is not None
```

### Documentación

- Documentar todas las funciones públicas
- Incluir ejemplos de uso
- Actualizar README si es necesario

```python
def new_function(param: str) -> dict:
    """
    Descripción de la función
    
    Args:
        param: Descripción del parámetro
        
    Returns:
        Descripción del retorno
        
    Example:
        >>> result = new_function("test")
        >>> print(result)
    """
    pass
```

## Proceso de Desarrollo

### 1. Setup

```bash
# Clonar
git clone <repo>
cd social_media_identity_clone_ai

# Crear entorno virtual
python -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Si existe
```

### 2. Desarrollo

```bash
# Crear rama
git checkout -b feature/my-feature

# Hacer cambios
# ...

# Tests
pytest

# Linting
flake8 .
black . --check

# Type checking
mypy .
```

### 3. Commit

Usar mensajes descriptivos:

```
feat: agregar sistema de notificaciones
fix: corregir error en validación de contenido
docs: actualizar documentación de API
refactor: mejorar estructura de servicios
test: agregar tests para ML service
```

### 4. Pull Request

- Descripción clara
- Referenciar issues relacionados
- Incluir screenshots si aplica
- Asegurar que tests pasen

## Áreas de Contribución

### Código
- Nuevas funcionalidades
- Corrección de bugs
- Optimizaciones
- Refactoring

### Documentación
- Mejorar documentación existente
- Agregar ejemplos
- Traducciones
- Tutoriales

### Tests
- Aumentar cobertura
- Tests de integración
- Tests de carga
- Tests de seguridad

### Diseño
- Mejoras de UX/UI
- Diseño de API
- Arquitectura

## Preguntas

Si tienes preguntas:
- Abre un issue
- Contacta al equipo
- Revisa documentación existente

## Código de Conducta

- Ser respetuoso
- Ser inclusivo
- Ser constructivo
- Ser profesional

## Reconocimiento

Los contribuidores serán reconocidos en:
- README.md
- CHANGELOG.md
- Releases

¡Gracias por contribuir! 🎉




