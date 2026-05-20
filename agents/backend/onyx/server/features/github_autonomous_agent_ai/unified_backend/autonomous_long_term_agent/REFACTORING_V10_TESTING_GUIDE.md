# Guía de Testing - Autonomous Long-Term Agent V10

## 📋 Resumen

Esta guía proporciona ejemplos detallados de cómo testear el código refactorizado V10, aprovechando las mejoras en testabilidad.

---

## 🎯 Ventajas de Testabilidad del Código Refactorizado V10

### Antes: Difícil de Testear

**Problemas:**
- ❌ Método largo con múltiples responsabilidades
- ❌ Difícil testear cada tipo de reflexión independientemente
- ❌ Difícil mockear dependencias específicas

### Después: Fácil de Testear

**Ventajas:**
- ✅ Métodos pequeños con responsabilidades únicas
- ✅ Fácil testear cada método independientemente
- ✅ Fácil mockear dependencias específicas

---

## 🧪 Ejemplos de Tests

### Test 1: `_should_run_reflection()`

```python
import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from autonomous_long_term_agent.core.periodic_tasks_coordinator import PeriodicTasksCoordinator

class TestShouldRunReflection(unittest.TestCase):
    
    def test_returns_true_when_no_last_reflection(self):
        """Test que retorna True cuando no hay última reflexión."""
        coordinator = PeriodicTasksCoordinator(
            health_checker=Mock(),
            metrics_manager=Mock(),
            self_reflection_engine=Mock(),
            task_queue=Mock(),
            agent_id="test_agent"
        )
        coordinator._last_reflection = None
        
        result = coordinator._should_run_reflection()
        
        self.assertTrue(result)
    
    def test_returns_false_when_interval_not_elapsed(self):
        """Test que retorna False cuando el intervalo no ha transcurrido."""
        coordinator = PeriodicTasksCoordinator(...)
        coordinator._last_reflection = datetime.utcnow() - timedelta(seconds=30)
        
        with patch('autonomous_long_term_agent.core.periodic_tasks_coordinator.settings') as mock_settings:
            mock_settings.self_reflection_interval = 60  # 60 segundos
            
            result = coordinator._should_run_reflection()
            
            self.assertFalse(result)
    
    def test_returns_true_when_interval_elapsed(self):
        """Test que retorna True cuando el intervalo ha transcurrido."""
        coordinator = PeriodicTasksCoordinator(...)
        coordinator._last_reflection = datetime.utcnow() - timedelta(seconds=120)
        
        with patch('autonomous_long_term_agent.core.periodic_tasks_coordinator.settings') as mock_settings:
            mock_settings.self_reflection_interval = 60  # 60 segundos
            
            result = coordinator._should_run_reflection()
            
            self.assertTrue(result)
```

**Beneficios:**
- ✅ Test simple y enfocado
- ✅ Fácil mockear settings
- ✅ Cubre todos los casos

---

### Test 2: `_get_recent_tasks_for_reflection()`

```python
class TestGetRecentTasksForReflection(unittest.IsolatedAsyncioTestCase):
    
    async def test_returns_empty_list_when_no_tasks(self):
        """Test que retorna lista vacía cuando no hay tareas."""
        mock_queue = Mock()
        mock_queue.get_recent_tasks = Mock(return_value=[])
        
        coordinator = PeriodicTasksCoordinator(
            health_checker=Mock(),
            metrics_manager=Mock(),
            self_reflection_engine=Mock(),
            task_queue=mock_queue,
            agent_id="test_agent"
        )
        
        result = await coordinator._get_recent_tasks_for_reflection()
        
        self.assertEqual(result, [])
        mock_queue.get_recent_tasks.assert_called_once_with(limit=10)
    
    async def test_returns_formatted_tasks_when_tasks_exist(self):
        """Test que retorna tareas formateadas cuando existen."""
        mock_queue = Mock()
        mock_tasks = [Mock(), Mock()]
        mock_queue.get_recent_tasks = Mock(return_value=mock_tasks)
        
        coordinator = PeriodicTasksCoordinator(...)
        
        with patch('autonomous_long_term_agent.core.periodic_tasks_coordinator.tasks_to_dict_list') as mock_format:
            mock_format.return_value = [{"id": "1"}, {"id": "2"}]
            
            result = await coordinator._get_recent_tasks_for_reflection()
            
            self.assertEqual(len(result), 2)
            mock_format.assert_called_once_with(mock_tasks)
```

**Beneficios:**
- ✅ Test async correcto
- ✅ Fácil mockear dependencias
- ✅ Verifica formato correcto

---

### Test 3: `_reflect_on_performance()`

```python
class TestReflectOnPerformance(unittest.IsolatedAsyncioTestCase):
    
    async def test_skips_when_disabled(self):
        """Test que omite cuando está deshabilitado."""
        mock_engine = Mock()
        coordinator = PeriodicTasksCoordinator(
            health_checker=Mock(),
            metrics_manager=Mock(),
            self_reflection_engine=mock_engine,
            task_queue=Mock(),
            agent_id="test_agent"
        )
        
        with patch('autonomous_long_term_agent.core.periodic_tasks_coordinator.settings') as mock_settings:
            mock_settings.self_reflection_on_performance = False
            
            await coordinator._reflect_on_performance({}, [])
            
            # ✅ Verificar que no se llamó el método de reflexión
            mock_engine.reflect_on_performance.assert_not_called()
    
    async def test_calls_reflection_when_enabled(self):
        """Test que llama reflexión cuando está habilitado."""
        mock_engine = Mock()
        coordinator = PeriodicTasksCoordinator(...)
        
        metrics = {"success_rate": 0.95}
        recent_tasks = [{"id": "1"}]
        
        with patch('autonomous_long_term_agent.core.periodic_tasks_coordinator.settings') as mock_settings:
            mock_settings.self_reflection_on_performance = True
            
            with patch('autonomous_long_term_agent.core.periodic_tasks_coordinator.safe_async_call') as mock_safe:
                await coordinator._reflect_on_performance(metrics, recent_tasks)
                
                # ✅ Verificar que se llamó con los parámetros correctos
                mock_safe.assert_called_once()
                call_args = mock_safe.call_args
                self.assertEqual(call_args[0][1], metrics)
                self.assertEqual(call_args[0][2], recent_tasks)
```

**Beneficios:**
- ✅ Test enfocado en un método específico
- ✅ Verifica comportamiento cuando está habilitado/deshabilitado
- ✅ Fácil mockear dependencias

---

### Test 4: `_perform_self_reflection()` - Integration Test

```python
class TestPerformSelfReflection(unittest.IsolatedAsyncioTestCase):
    
    async def test_complete_reflection_flow(self):
        """Test flujo completo de reflexión."""
        mock_engine = Mock()
        mock_queue = Mock()
        mock_queue.get_recent_tasks = Mock(return_value=[])
        
        coordinator = PeriodicTasksCoordinator(
            health_checker=Mock(),
            metrics_manager=Mock(),
            self_reflection_engine=mock_engine,
            task_queue=mock_queue,
            agent_id="test_agent"
        )
        
        # ✅ Setup: Configurar que debe ejecutarse
        coordinator._last_reflection = None
        
        with patch('autonomous_long_term_agent.core.periodic_tasks_coordinator.settings') as mock_settings:
            mock_settings.self_reflection_on_performance = True
            mock_settings.self_reflection_on_capabilities = True
            mock_settings.self_reflection_interval = 60
            
            with patch.object(coordinator, '_reflect_on_performance') as mock_perf:
                with patch.object(coordinator, '_reflect_on_capabilities') as mock_cap:
                    with patch.object(coordinator, '_perform_periodic_reflection') as mock_periodic:
                        await coordinator._perform_self_reflection()
                        
                        # ✅ Verificar que todos los métodos fueron llamados
                        mock_perf.assert_called_once()
                        mock_cap.assert_called_once()
                        mock_periodic.assert_called_once()
                        
                        # ✅ Verificar que se actualizó el timestamp
                        self.assertIsNotNone(coordinator._last_reflection)
    
    async def test_skips_when_engine_not_available(self):
        """Test que omite cuando el engine no está disponible."""
        coordinator = PeriodicTasksCoordinator(
            health_checker=Mock(),
            metrics_manager=Mock(),
            self_reflection_engine=None,  # ✅ Sin engine
            task_queue=Mock(),
            agent_id="test_agent"
        )
        
        with patch.object(coordinator, '_should_run_reflection') as mock_should:
            await coordinator._perform_self_reflection()
            
            # ✅ Verificar que no se llamó ningún método
            mock_should.assert_not_called()
```

**Beneficios:**
- ✅ Test de integración completo
- ✅ Verifica flujo end-to-end
- ✅ Mockea dependencias apropiadamente

---

### Test 5: `_perform_world_based_planning()`

```python
class TestPerformWorldBasedPlanning(unittest.IsolatedAsyncioTestCase):
    
    async def test_creates_plan_when_world_model_available(self):
        """Test que crea plan cuando el world model está disponible."""
        mock_world_model = Mock()
        mock_world_model.plan_based_on_world = Mock(return_value={
            "planning_strategy": "explore",
            "recommended_actions": ["action1", "action2"]
        })
        
        handler = AutonomousOperationHandler(
            learning_engine=Mock(),
            world_model=mock_world_model,
            agent_id="test_agent",
            instruction="Test instruction"
        )
        
        with patch('autonomous_long_term_agent.core.autonomous_operation_handler.safe_async_call') as mock_safe:
            mock_safe.return_value = {
                "planning_strategy": "explore",
                "recommended_actions": ["action1", "action2"]
            }
            
            await handler._perform_world_based_planning()
            
            # ✅ Verificar que se llamó con el goal correcto
            mock_safe.assert_called_once()
            call_args = mock_safe.call_args
            self.assertEqual(call_args[0][1], "Test instruction")
    
    async def test_handles_none_world_model(self):
        """Test que maneja cuando world_model es None."""
        handler = AutonomousOperationHandler(
            learning_engine=Mock(),
            world_model=None,  # ✅ Sin world model
            agent_id="test_agent",
            instruction="Test instruction"
        )
        
        # ✅ No debe lanzar error
        await handler._perform_world_based_planning()
```

**Beneficios:**
- ✅ Test enfocado en un método específico
- ✅ Verifica comportamiento con/sin world model
- ✅ Fácil mockear dependencias

---

## 🎯 Estrategias de Testing

### 1. Unit Tests - Métodos Individuales

**Estrategia**: Testear cada método helper independientemente

```python
# ✅ Test método individual
def test_should_run_reflection():
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
async def test_perform_self_reflection_complete_flow():
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
async def test_skips_when_engine_not_available():
    # Test sin engine
    
async def test_skips_when_interval_not_elapsed():
    # Test cuando intervalo no ha transcurrido
```

**Beneficios:**
- ✅ Robustez
- ✅ Manejo de errores
- ✅ Casos reales

---

## ✅ Resumen de Testing

### Ventajas del Código Refactorizado V10

1. ✅ **Métodos pequeños**: Fácil testear
2. ✅ **SRP aplicado**: Tests enfocados
3. ✅ **Dependencias mockeables**: Fácil mockear
4. ✅ **Type hints**: Mejor IDE support

### Cobertura de Tests Posible

| Método | Testabilidad | Complejidad |
|--------|--------------|-------------|
| `_should_run_reflection()` | ✅ Alta | Baja |
| `_get_recent_tasks_for_reflection()` | ✅ Alta | Baja |
| `_reflect_on_performance()` | ✅ Alta | Baja |
| `_reflect_on_capabilities()` | ✅ Alta | Baja |
| `_perform_periodic_reflection()` | ✅ Alta | Baja |
| `_perform_self_reflection()` | ✅ Alta | Media |

---

**🎊🎊🎊 Guía de Testing Completa. Código Altamente Testeable. 🎊🎊🎊**

