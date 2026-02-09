# Guía de Testing - Audio Separator Refactorizado

## 📋 Resumen

Esta guía proporciona ejemplos detallados de cómo testear el código refactorizado, aprovechando las mejoras en testabilidad.

---

## 🎯 Ventajas de Testabilidad del Código Refactorizado

### Antes: Difícil de Testear

**Problemas:**
- ❌ Métodos faltantes imposibles de testear
- ❌ Métodos largos con múltiples responsabilidades
- ❌ Dependencias acopladas
- ❌ Difícil mockear

### Después: Fácil de Testear

**Ventajas:**
- ✅ Métodos pequeños con responsabilidades únicas
- ✅ Dependencias inyectadas (fácil mockear)
- ✅ Métodos separados (fácil testear independientemente)
- ✅ Type hints (mejor IDE support)

---

## 🧪 Ejemplos de Tests

### Test 1: `_try_model_separate_method()`

```python
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from audio_separator.separator import AudioSeparator

class TestTryModelSeparateMethod(unittest.TestCase):
    
    def test_success_when_model_has_separate_method(self):
        """Test éxito cuando modelo tiene método separate."""
        separator = AudioSeparator(model_type="demucs")
        
        # ✅ Mock modelo con método separate
        mock_model = MagicMock()
        mock_model.separate = Mock(return_value={
            'vocals': 'output/vocals.wav',
            'drums': 'output/drums.wav'
        })
        separator.model = mock_model
        
        # Test
        result = separator._try_model_separate_method(
            Path("test.wav"),
            Path("output/")
        )
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('vocals', result)
        self.assertIn('drums', result)
        mock_model.separate.assert_called_once_with(
            "test.wav",
            "output/"
        )
    
    def test_returns_none_when_model_is_none(self):
        """Test retorna None cuando modelo es None."""
        separator = AudioSeparator(model_type="demucs")
        separator.model = None
        
        result = separator._try_model_separate_method(
            Path("test.wav"),
            Path("output/")
        )
        
        self.assertIsNone(result)
    
    def test_returns_none_when_model_no_separate_method(self):
        """Test retorna None cuando modelo no tiene método separate."""
        separator = AudioSeparator(model_type="demucs")
        
        # ✅ Mock modelo sin método separate
        mock_model = MagicMock()
        del mock_model.separate  # Remover método
        separator.model = mock_model
        
        result = separator._try_model_separate_method(
            Path("test.wav"),
            Path("output/")
        )
        
        self.assertIsNone(result)
    
    def test_fallback_on_exception(self):
        """Test fallback cuando método separate falla."""
        separator = AudioSeparator(model_type="demucs")
        
        # ✅ Mock modelo que lanza excepción
        mock_model = MagicMock()
        mock_model.separate = Mock(side_effect=Exception("Model error"))
        separator.model = mock_model
        
        result = separator._try_model_separate_method(
            Path("test.wav"),
            Path("output/")
        )
        
        # ✅ Debe retornar None para fallback
        self.assertIsNone(result)
```

**Beneficios:**
- ✅ Tests simples y enfocados
- ✅ Fácil mockear dependencias
- ✅ Cubre todos los casos

---

### Test 2: `_perform_separation_pipeline()`

```python
class TestPerformSeparationPipeline(unittest.TestCase):
    
    def test_complete_pipeline_success(self):
        """Test pipeline completo exitoso."""
        separator = AudioSeparator(model_type="demucs")
        
        # ✅ Mock todos los componentes
        mock_loader = Mock()
        mock_loader.load = Mock(return_value=(
            np.array([0.1, 0.2, 0.3]),  # audio
            44100  # sample_rate
        ))
        separator.loader = mock_loader
        
        mock_preprocessor = Mock()
        mock_preprocessor.process = Mock(return_value=torch.tensor([[0.1, 0.2, 0.3]]))
        separator.preprocessor = mock_preprocessor
        
        mock_model = Mock()
        mock_model.forward = Mock(return_value={
            'vocals': torch.tensor([[0.1, 0.2]]),
            'drums': torch.tensor([[0.1, 0.2]])
        })
        separator.model = mock_model
        
        mock_postprocessor = Mock()
        mock_postprocessor.process = Mock(return_value={
            'vocals': np.array([0.1, 0.2]),
            'drums': np.array([0.1, 0.2])
        })
        separator.postprocessor = mock_postprocessor
        
        # Test
        result = separator._perform_separation_pipeline(Path("test.wav"))
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn('vocals', result)
        self.assertIn('drums', result)
        
        # ✅ Verificar que cada componente fue llamado
        mock_loader.load.assert_called_once()
        mock_preprocessor.process.assert_called_once()
        mock_model.forward.assert_called_once()
        mock_postprocessor.process.assert_called_once()
    
    def test_pipeline_failure_raises_error(self):
        """Test que pipeline lanza error cuando falla."""
        separator = AudioSeparator(model_type="demucs")
        
        # ✅ Mock loader que falla
        mock_loader = Mock()
        mock_loader.load = Mock(side_effect=Exception("Load error"))
        separator.loader = mock_loader
        
        # Test
        with self.assertRaises(AudioProcessingError) as context:
            separator._perform_separation_pipeline(Path("test.wav"))
        
        # ✅ Verificar error específico
        self.assertIn("PIPELINE_FAILED", str(context.exception))
        self.assertIn("test.wav", str(context.exception))
```

**Beneficios:**
- ✅ Test completo del pipeline
- ✅ Fácil mockear cada componente
- ✅ Verifica flujo completo

---

### Test 3: `_save_separated_sources()`

```python
class TestSaveSeparatedSources(unittest.TestCase):
    
    def test_save_all_sources_success(self):
        """Test guardar todas las fuentes exitosamente."""
        separator = AudioSeparator(model_type="demucs")
        
        # ✅ Mock saver
        mock_saver = Mock()
        separator.saver = mock_saver
        
        separated_audio = {
            'vocals': np.array([0.1, 0.2, 0.3]),
            'drums': np.array([0.1, 0.2, 0.3])
        }
        
        # Test
        result = separator._save_separated_sources(
            separated_audio,
            Path("song.wav"),
            Path("output/"),
            save_outputs=True
        )
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertIn('vocals', result)
        self.assertIn('drums', result)
        
        # ✅ Verificar que saver fue llamado para cada fuente
        self.assertEqual(mock_saver.save.call_count, 2)
    
    def test_save_outputs_false_returns_none(self):
        """Test que no guarda cuando save_outputs=False."""
        separator = AudioSeparator(model_type="demucs")
        mock_saver = Mock()
        separator.saver = mock_saver
        
        separated_audio = {
            'vocals': np.array([0.1, 0.2, 0.3])
        }
        
        result = separator._save_separated_sources(
            separated_audio,
            Path("song.wav"),
            Path("output/"),
            save_outputs=False  # ✅ No guardar
        )
        
        # ✅ Verificar que no se llamó saver
        mock_saver.save.assert_not_called()
        
        # ✅ Verificar que retorna None
        self.assertEqual(result['vocals'], None)
    
    def test_handles_save_failure_gracefully(self):
        """Test que maneja fallos de guardado gracefully."""
        separator = AudioSeparator(model_type="demucs")
        
        # ✅ Mock saver que falla
        mock_saver = Mock()
        mock_saver.save = Mock(side_effect=Exception("Save error"))
        separator.saver = mock_saver
        
        separated_audio = {
            'vocals': np.array([0.1, 0.2, 0.3])
        }
        
        # Test - debe lanzar error
        with self.assertRaises(AudioIOError):
            separator._save_separated_sources(
                separated_audio,
                Path("song.wav"),
                Path("output/"),
                save_outputs=True
            )
```

**Beneficios:**
- ✅ Tests de casos exitosos y fallos
- ✅ Verifica comportamiento con save_outputs
- ✅ Manejo de errores testado

---

### Test 4: Normalización Centralizada

```python
class TestNormalizeAudio(unittest.TestCase):
    
    def test_normalize_always_when_check_clipping_false(self):
        """Test que siempre normaliza cuando check_clipping=False."""
        from audio_separator.processor import AudioPreprocessor
        
        preprocessor = AudioPreprocessor()
        audio = np.array([0.5, 0.6, 0.7])  # Max = 0.7
        
        result = preprocessor._normalize(audio)
        
        # ✅ Debe normalizar siempre (max debería ser 1.0)
        self.assertAlmostEqual(np.abs(result).max(), 1.0, places=5)
    
    def test_normalize_only_when_clipping_when_check_clipping_true(self):
        """Test que solo normaliza si hay clipping cuando check_clipping=True."""
        from audio_separator.processor import AudioPostprocessor
        
        postprocessor = AudioPostprocessor()
        
        # ✅ Audio sin clipping (max < 1.0)
        audio_no_clip = np.array([0.5, 0.6, 0.7])
        result_no_clip = postprocessor._normalize(audio_no_clip)
        
        # ✅ No debe normalizar (debe mantener valores originales)
        np.testing.assert_array_almost_equal(result_no_clip, audio_no_clip)
        
        # ✅ Audio con clipping (max > 1.0)
        audio_clip = np.array([0.5, 1.2, 0.7])  # Max = 1.2
        result_clip = postprocessor._normalize(audio_clip)
        
        # ✅ Debe normalizar (max debería ser 1.0)
        self.assertAlmostEqual(np.abs(result_clip).max(), 1.0, places=5)
    
    def test_normalize_handles_zero_max(self):
        """Test que maneja audio con max=0."""
        from audio_separator.processor.base_processor import BaseAudioProcessor
        
        processor = BaseAudioProcessor()
        audio = np.array([0.0, 0.0, 0.0])
        
        result = processor._normalize_audio(audio, check_clipping=False)
        
        # ✅ No debe cambiar (división por cero evitada)
        np.testing.assert_array_equal(result, audio)
```

**Beneficios:**
- ✅ Tests de lógica de normalización
- ✅ Verifica diferentes comportamientos
- ✅ Casos edge testados

---

### Test 5: Integration Test Completo

```python
class TestAudioSeparatorIntegration(unittest.TestCase):
    
    @patch('audio_separator.separator.audio_separator.AudioLoader')
    @patch('audio_separator.separator.audio_separator.AudioSaver')
    @patch('audio_separator.separator.audio_separator.build_audio_separator_model')
    def test_separate_file_complete_flow(self, mock_build_model, mock_saver_class, mock_loader_class):
        """Test flujo completo de separate_file."""
        # ✅ Setup mocks
        mock_model = MagicMock()
        mock_model.separate = None  # No tiene método separate
        mock_model.forward = Mock(return_value={
            'vocals': torch.tensor([[0.1, 0.2]]),
            'drums': torch.tensor([[0.1, 0.2]])
        })
        mock_build_model.return_value = mock_model
        
        mock_loader = Mock()
        mock_loader.load = Mock(return_value=(
            np.array([0.1, 0.2, 0.3]),
            44100
        ))
        mock_loader_class.return_value = mock_loader
        
        mock_saver = Mock()
        mock_saver_class.return_value = mock_saver
        
        # ✅ Crear separator
        separator = AudioSeparator(model_type="demucs")
        separator.loader = mock_loader
        separator.saver = mock_saver
        
        # Test
        result = separator.separate_file(
            "test.wav",
            "output/",
            save_outputs=True
        )
        
        # ✅ Verificar resultado
        self.assertIsInstance(result, dict)
        self.assertIn('vocals', result)
        self.assertIn('drums', result)
        
        # ✅ Verificar que se llamaron los métodos correctos
        mock_loader.load.assert_called_once()
        mock_model.forward.assert_called_once()
        self.assertEqual(mock_saver.save.call_count, 2)  # Una vez por fuente
```

**Beneficios:**
- ✅ Test de integración completo
- ✅ Verifica flujo end-to-end
- ✅ Mockea dependencias externas

---

## 🎯 Estrategias de Testing

### 1. Unit Tests - Métodos Individuales

**Estrategia**: Testear cada método independientemente

```python
# ✅ Test método individual
def test_try_model_separate_method():
    # Mock dependencias
    # Test método
    # Assert resultado
```

**Beneficios:**
- ✅ Tests rápidos
- ✅ Tests enfocados
- ✅ Fácil identificar problemas

---

### 2. Integration Tests - Flujo Completo

**Estrategia**: Testear flujo completo con mocks

```python
# ✅ Test flujo completo
def test_separate_file_complete_flow():
    # Mock todas las dependencias
    # Test flujo completo
    # Assert resultado final
```

**Beneficios:**
- ✅ Verifica integración
- ✅ Detecta problemas de flujo
- ✅ Más realista

---

### 3. Edge Cases - Casos Límite

**Estrategia**: Testear casos edge y errores

```python
# ✅ Test casos edge
def test_empty_audio():
    # Test con audio vacío
    
def test_invalid_path():
    # Test con path inválido
    
def test_model_failure():
    # Test cuando modelo falla
```

**Beneficios:**
- ✅ Robustez
- ✅ Manejo de errores
- ✅ Casos reales

---

## ✅ Resumen de Testing

### Ventajas del Código Refactorizado

1. ✅ **Métodos pequeños**: Fácil testear
2. ✅ **Dependencias inyectadas**: Fácil mockear
3. ✅ **SRP aplicado**: Tests enfocados
4. ✅ **Type hints**: Mejor IDE support

### Cobertura de Tests Posible

| Método | Testabilidad | Complejidad |
|--------|--------------|-------------|
| `_try_model_separate_method()` | ✅ Alta | Baja |
| `_perform_separation_pipeline()` | ✅ Alta | Media |
| `_save_separated_sources()` | ✅ Alta | Baja |
| `_normalize_audio()` | ✅ Alta | Baja |

---

**🎊🎊🎊 Guía de Testing Completa. Código Altamente Testeable. 🎊🎊🎊**

