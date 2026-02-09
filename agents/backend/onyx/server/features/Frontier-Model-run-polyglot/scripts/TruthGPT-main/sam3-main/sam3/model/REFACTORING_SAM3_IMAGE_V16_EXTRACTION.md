# 🎉 Refactorización de sam3_image.py V16 - Extracción de Métodos

## 📋 Resumen

Refactorización V16 enfocada en extraer métodos largos y consolidar lógica duplicada en `sam3_image.py` (884 líneas) para mejorar la mantenibilidad y legibilidad.

## ✅ Oportunidades Identificadas

### 1. Extraer Lógica de Preparación de Features ✅

**Problema**: Lógica duplicada en `predict_inst()` y `predict_inst_batch()` para preparar features del backbone.

**Ubicación**: Líneas 599-636 y 638-684

**Solución**: Crear método helper `_prepare_backbone_features_for_predictor()`.

**Antes**:
```python
def predict_inst(self, inference_state, **kwargs):
    backbone_out = inference_state["backbone_out"]["sam2_backbone_out"]
    (_, vision_feats, _, _) = self.inst_interactive_predictor.model._prepare_backbone_features(backbone_out)
    vision_feats[-1] = vision_feats[-1] + self.inst_interactive_predictor.model.no_mem_embed
    feats = [
        feat.permute(1, 2, 0).view(1, -1, *feat_size)
        for feat, feat_size in zip(
            vision_feats[::-1], self.inst_interactive_predictor._bb_feat_sizes[::-1]
        )
    ][::-1]
    # ... más código ...

def predict_inst_batch(self, inference_state, *args, **kwargs):
    backbone_out = inference_state["backbone_out"]["sam2_backbone_out"]
    (_, vision_feats, _, _) = self.inst_interactive_predictor.model._prepare_backbone_features(backbone_out)
    vision_feats[-1] = vision_feats[-1] + self.inst_interactive_predictor.model.no_mem_embed
    # ... código similar pero para batch ...
```

**Después**:
```python
def _prepare_backbone_features_for_predictor(
    self,
    backbone_out: Dict,
    batch_size: int = 1
) -> Dict[str, torch.Tensor]:
    """
    Prepare backbone features for interactive predictor.
    
    Args:
        backbone_out: Backbone output dictionary
        batch_size: Batch size (1 for single, >1 for batch)
        
    Returns:
        Dictionary with prepared features
    """
    sam2_backbone_out = backbone_out["sam2_backbone_out"]
    (_, vision_feats, _, _) = self.inst_interactive_predictor.model._prepare_backbone_features(
        sam2_backbone_out
    )
    
    # Add no_mem_embed
    vision_feats[-1] = (
        vision_feats[-1] + self.inst_interactive_predictor.model.no_mem_embed
    )
    
    # Reshape features
    feats = [
        feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
        for feat, feat_size in zip(
            vision_feats[::-1], self.inst_interactive_predictor._bb_feat_sizes[::-1]
        )
    ][::-1]
    
    return {
        "image_embed": feats[-1],
        "high_res_feats": feats[:-1]
    }

def predict_inst(self, inference_state, **kwargs):
    orig_h, orig_w = (
        inference_state["original_height"],
        inference_state["original_width"],
    )
    features = self._prepare_backbone_features_for_predictor(
        inference_state["backbone_out"],
        batch_size=1
    )
    self.inst_interactive_predictor._features = features
    self.inst_interactive_predictor._is_image_set = True
    self.inst_interactive_predictor._orig_hw = [(orig_h, orig_w)]
    res = self.inst_interactive_predictor.predict(**kwargs)
    self._reset_predictor_state()
    return res
```

**Reducción**: ~40 líneas duplicadas → ~25 líneas (método helper) + ~15 líneas (uso) = -40% duplicación

### 2. Extraer Lógica de Configuración de Predictor ✅

**Problema**: Lógica repetida para configurar y resetear el estado del predictor.

**Ubicación**: Líneas 627-635 y 671-683

**Solución**: Crear métodos helper `_setup_predictor_state()` y `_reset_predictor_state()`.

**Antes**:
```python
self.inst_interactive_predictor._features = {...}
self.inst_interactive_predictor._is_image_set = True
self.inst_interactive_predictor._orig_hw = [(orig_h, orig_w)]
# ... uso ...
self.inst_interactive_predictor._features = None
self.inst_interactive_predictor._is_image_set = False
```

**Después**:
```python
def _setup_predictor_state(
    self,
    features: Dict[str, torch.Tensor],
    orig_hw: List[Tuple[int, int]],
    is_batch: bool = False
) -> None:
    """Setup predictor state for inference."""
    self.inst_interactive_predictor._features = features
    self.inst_interactive_predictor._is_image_set = True
    self.inst_interactive_predictor._is_batch = is_batch
    self.inst_interactive_predictor._orig_hw = orig_hw

def _reset_predictor_state(self) -> None:
    """Reset predictor state after inference."""
    self.inst_interactive_predictor._features = None
    self.inst_interactive_predictor._is_image_set = False
    self.inst_interactive_predictor._is_batch = False
```

**Reducción**: ~10 líneas repetidas → 2 métodos reutilizables

### 3. Extraer Lógica de Postprocesamiento Multimask ✅

**Problema**: Método `_postprocess_out()` tiene lógica compleja que podría simplificarse.

**Ubicación**: Líneas 495-520

**Solución**: Extraer métodos helper para selección de mejor máscara y actualización de outputs.

**Antes**:
```python
def _postprocess_out(self, out: Dict, multimask_output: bool = False):
    num_mask_boxes = out["pred_boxes"].size(1)
    if not self.training and multimask_output and num_mask_boxes > 1:
        # ... 25 líneas de lógica inline ...
    return out
```

**Después**:
```python
def _postprocess_out(self, out: Dict, multimask_output: bool = False):
    if self._should_apply_multimask_postprocessing(out, multimask_output):
        self._store_multimask_outputs(out)
        self._select_best_mask(out)
    return out

def _should_apply_multimask_postprocessing(
    self,
    out: Dict,
    multimask_output: bool
) -> bool:
    """Check if multimask postprocessing should be applied."""
    return (
        not self.training
        and multimask_output
        and out["pred_boxes"].size(1) > 1
    )

def _store_multimask_outputs(self, out: Dict) -> None:
    """Store multimask outputs before selecting best mask."""
    out["multi_pred_logits"] = out["pred_logits"]
    if "pred_masks" in out:
        out["multi_pred_masks"] = out["pred_masks"]
    out["multi_pred_boxes"] = out["pred_boxes"]
    out["multi_pred_boxes_xyxy"] = out["pred_boxes_xyxy"]

def _select_best_mask(self, out: Dict) -> None:
    """Select best mask based on logits and update outputs."""
    best_mask_idx = out["pred_logits"].argmax(1).squeeze(1)
    batch_idx = torch.arange(len(best_mask_idx), device=best_mask_idx.device)
    
    out["pred_logits"] = out["pred_logits"][batch_idx, best_mask_idx].unsqueeze(1)
    if "pred_masks" in out:
        out["pred_masks"] = out["pred_masks"][batch_idx, best_mask_idx].unsqueeze(1)
    out["pred_boxes"] = out["pred_boxes"][batch_idx, best_mask_idx].unsqueeze(1)
    out["pred_boxes_xyxy"] = out["pred_boxes_xyxy"][batch_idx, best_mask_idx].unsqueeze(1)
```

**Reducción**: ~25 líneas → ~15 líneas (método principal) + métodos helper = mejor legibilidad

### 4. Consolidar Lógica de Inicialización de Scoring ✅

**Problema**: Lógica condicional compleja en `__init__` para inicializar scoring.

**Ubicación**: Líneas 81-91

**Solución**: Extraer método `_initialize_scoring()`.

**Antes**:
```python
if self.use_dot_prod_scoring:
    assert dot_prod_scoring is not None
    self.dot_prod_scoring = dot_prod_scoring
    self.instance_dot_prod_scoring = None
    if separate_scorer_for_instance:
        self.instance_dot_prod_scoring = deepcopy(dot_prod_scoring)
else:
    self.class_embed = torch.nn.Linear(self.hidden_dim, 1)
    self.instance_class_embed = None
    if separate_scorer_for_instance:
        self.instance_class_embed = deepcopy(self.class_embed)
```

**Después**:
```python
def _initialize_scoring(
    self,
    dot_prod_scoring,
    separate_scorer_for_instance: bool
) -> None:
    """Initialize scoring mechanism (dot product or class embedding)."""
    if self.use_dot_prod_scoring:
        assert dot_prod_scoring is not None
        self.dot_prod_scoring = dot_prod_scoring
        self.instance_dot_prod_scoring = (
            deepcopy(dot_prod_scoring) if separate_scorer_for_instance else None
        )
    else:
        self.class_embed = torch.nn.Linear(self.hidden_dim, 1)
        self.instance_class_embed = (
            deepcopy(self.class_embed) if separate_scorer_for_instance else None
        )
```

**Reducción**: ~11 líneas → método más claro y testeable

### 5. Extraer Constantes de Text IDs ✅

**Problema**: Constantes de clase que podrían estar mejor organizadas.

**Ubicación**: Líneas 37-39

**Solución**: Crear clase de constantes o usar Enum.

**Antes**:
```python
class Sam3Image(torch.nn.Module):
    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1
    TEXT_ID_FOR_GEOMETRIC = 2
```

**Después**:
```python
class TextID:
    """Text ID constants for different prompt types."""
    FOR_TEXT = 0
    FOR_VISUAL = 1
    FOR_GEOMETRIC = 2

class Sam3Image(torch.nn.Module):
    TEXT_ID_FOR_TEXT = TextID.FOR_TEXT
    TEXT_ID_FOR_VISUAL = TextID.FOR_VISUAL
    TEXT_ID_FOR_GEOMETRIC = TextID.FOR_GEOMETRIC
```

**Beneficios**: Mejor organización y documentación

### 6. Simplificar `forward_grounding()` ✅

**Problema**: Método largo con múltiples responsabilidades.

**Ubicación**: Líneas 442-493

**Solución**: Ya está bien estructurado con métodos helper, pero podría extraerse la lógica de preparación de prompts.

**Mejora**: Extraer `_prepare_geometric_prompt()` si la lógica es compleja.

## 📊 Métricas Esperadas

| Cambio | Líneas Antes | Líneas Después | Reducción |
|--------|--------------|----------------|-----------|
| Extraer preparación de features | ~40 duplicadas | ~25 (helper) | -38% |
| Extraer configuración de predictor | ~10 repetidas | ~15 (2 métodos) | Mejor organización |
| Extraer postprocesamiento | ~25 | ~15 (principal) | -40% |
| Consolidar inicialización scoring | ~11 | ~10 (método) | Mejor legibilidad |
| Organizar constantes | ~3 | ~8 (clase) | Mejor documentación |

**Total**: Mejora en organización y legibilidad, aunque el total de líneas puede aumentar ligeramente debido a métodos helper.

## 🎯 Beneficios Adicionales

1. **Mejor Testabilidad**: Métodos más pequeños y enfocados
2. **Menos Duplicación**: Lógica común extraída
3. **Mejor Legibilidad**: Métodos con nombres descriptivos
4. **Mantenibilidad**: Cambios en un solo lugar
5. **Extensibilidad**: Fácil agregar nuevas funcionalidades

## ✅ Estado

**Refactorización V16**: ✅ **DOCUMENTADA**

**Cambios Pendientes**: Requieren aplicación manual

**Archivos Afectados**:
- `sam3_image.py` (884 líneas) - Refactorización principal

