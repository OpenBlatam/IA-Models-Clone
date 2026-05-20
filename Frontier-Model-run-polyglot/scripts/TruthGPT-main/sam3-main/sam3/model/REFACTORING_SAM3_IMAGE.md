# Refactorización de sam3_image.py

## 📋 Resumen

Refactorización de `sam3_image.py` aplicando principios SOLID, DRY y mejores prácticas, siguiendo el mismo enfoque usado en otros módulos.

## 🎯 Mejoras Aplicadas

### 1. **Métodos Helper Extraídos para Selección de Heads**

**Antes:**
```python
def _update_scores_and_boxes(...):
    # ... código ...
    # score prediction
    if self.use_dot_prod_scoring:
        dot_prod_scoring_head = self.dot_prod_scoring
        if is_instance_prompt and self.instance_dot_prod_scoring is not None:
            dot_prod_scoring_head = self.instance_dot_prod_scoring
        outputs_class = dot_prod_scoring_head(hs, prompt, prompt_mask)
    else:
        class_embed_head = self.class_embed
        if is_instance_prompt and self.instance_class_embed is not None:
            class_embed_head = self.instance_class_embed
        outputs_class = class_embed_head(hs)

    # box prediction
    box_head = self.transformer.decoder.bbox_embed
    if (
        is_instance_prompt
        and self.transformer.decoder.instance_bbox_embed is not None
    ):
        box_head = self.transformer.decoder.instance_bbox_embed
```

**Después:**
```python
def _get_scoring_head(self, is_instance_prompt: bool):
    """Get the appropriate scoring head based on configuration."""
    if self.use_dot_prod_scoring:
        head = self.dot_prod_scoring
        if is_instance_prompt and self.instance_dot_prod_scoring is not None:
            head = self.instance_dot_prod_scoring
        return head
    else:
        head = self.class_embed
        if is_instance_prompt and self.instance_class_embed is not None:
            head = self.instance_class_embed
        return head

def _compute_class_scores(self, scoring_head, hs, prompt, prompt_mask):
    """Compute class scores using the given scoring head."""
    if self.use_dot_prod_scoring:
        return scoring_head(hs, prompt, prompt_mask)
    else:
        return scoring_head(hs)

def _get_box_head(self, is_instance_prompt: bool):
    """Get the appropriate box prediction head."""
    box_head = self.transformer.decoder.bbox_embed
    if (
        is_instance_prompt
        and self.transformer.decoder.instance_bbox_embed is not None
    ):
        box_head = self.transformer.decoder.instance_bbox_embed
    return box_head

def _update_scores_and_boxes(...):
    # ... código ...
    scoring_head = self._get_scoring_head(is_instance_prompt)
    outputs_class = self._compute_class_scores(
        scoring_head, hs, prompt, prompt_mask
    )
    box_head = self._get_box_head(is_instance_prompt)
```

**Beneficios:**
- ✅ Métodos pequeños y enfocados (SRP)
- ✅ Lógica reutilizable
- ✅ Más fácil de testear
- ✅ Código más legible

### 2. **Métodos Helper Extraídos para Postprocesamiento de Multimask**

**Antes:**
```python
def _postprocess_out(self, out: Dict, multimask_output: bool = False):
    # For multimask output, during eval we return the single best mask...
    num_mask_boxes = out["pred_boxes"].size(1)
    if not self.training and multimask_output and num_mask_boxes > 1:
        out["multi_pred_logits"] = out["pred_logits"]
        if "pred_masks" in out:
            out["multi_pred_masks"] = out["pred_masks"]
        out["multi_pred_boxes"] = out["pred_boxes"]
        out["multi_pred_boxes_xyxy"] = out["pred_boxes_xyxy"]

        best_mask_idx = out["pred_logits"].argmax(1).squeeze(1)
        batch_idx = torch.arange(len(best_mask_idx), device=best_mask_idx.device)

        out["pred_logits"] = out["pred_logits"][batch_idx, best_mask_idx].unsqueeze(1)
        if "pred_masks" in out:
            out["pred_masks"] = out["pred_masks"][batch_idx, best_mask_idx].unsqueeze(1)
        out["pred_boxes"] = out["pred_boxes"][batch_idx, best_mask_idx].unsqueeze(1)
        out["pred_boxes_xyxy"] = out["pred_boxes_xyxy"][batch_idx, best_mask_idx].unsqueeze(1)

    return out
```

**Después:**
```python
def _should_postprocess_multimask(self, out: Dict, multimask_output: bool) -> bool:
    """Check if multimask postprocessing should be applied."""
    return (
        not self.training
        and multimask_output
        and out["pred_boxes"].size(1) > 1
    )

def _save_multimask_outputs(self, out: Dict) -> None:
    """Save multimask outputs with 'multi_' prefix."""
    out["multi_pred_logits"] = out["pred_logits"]
    if "pred_masks" in out:
        out["multi_pred_masks"] = out["pred_masks"]
    out["multi_pred_boxes"] = out["pred_boxes"]
    out["multi_pred_boxes_xyxy"] = out["pred_boxes_xyxy"]

def _select_best_mask(self, out: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select the best mask index for each item in the batch."""
    best_mask_idx = out["pred_logits"].argmax(1).squeeze(1)
    batch_idx = torch.arange(len(best_mask_idx), device=best_mask_idx.device)
    return best_mask_idx, batch_idx

def _apply_best_mask_selection(
    self, out: Dict, best_mask_idx: torch.Tensor, batch_idx: torch.Tensor
) -> None:
    """Apply best mask selection to output dictionary."""
    out["pred_logits"] = out["pred_logits"][batch_idx, best_mask_idx].unsqueeze(1)
    if "pred_masks" in out:
        out["pred_masks"] = out["pred_masks"][batch_idx, best_mask_idx].unsqueeze(1)
    out["pred_boxes"] = out["pred_boxes"][batch_idx, best_mask_idx].unsqueeze(1)
    out["pred_boxes_xyxy"] = out["pred_boxes_xyxy"][batch_idx, best_mask_idx].unsqueeze(1)

def _postprocess_out(self, out: Dict, multimask_output: bool = False):
    """Postprocess output for multimask evaluation."""
    if self._should_postprocess_multimask(out, multimask_output):
        self._save_multimask_outputs(out)
        best_mask_idx, batch_idx = self._select_best_mask(out)
        self._apply_best_mask_selection(out, best_mask_idx, batch_idx)

    return out
```

**Beneficios:**
- ✅ Métodos pequeños y enfocados (SRP)
- ✅ Lógica clara y separada
- ✅ Más fácil de testear
- ✅ Mejor documentación

### 3. **Reemplazo de Print por Warnings**

**Antes:**
```python
if find_input.input_points is not None and find_input.input_points.numel() > 0:
    print("Warning: Point prompts are ignored in PCS.")
```

**Después:**
```python
if find_input.input_points is not None and find_input.input_points.numel() > 0:
    import warnings
    warnings.warn("Point prompts are ignored in PCS.", UserWarning)
```

**Beneficios:**
- ✅ Uso apropiado de warnings de Python
- ✅ Mejor control de advertencias
- ✅ Más profesional

## 📊 Métricas

| Aspecto | Antes | Después | Mejora |
|---------|-------|---------|--------|
| **Métodos helper** | 0 | 7 | **+7** |
| **Print statements** | 1 | 0 | **-100%** |
| **Warnings** | 0 | 1 | **+1** |
| **Líneas en `_update_scores_and_boxes`** | ~85 | ~60 | **-29%** |
| **Líneas en `_postprocess_out`** | ~25 | ~10 | **-60%** |

## ✅ Principios Aplicados

### DRY (Don't Repeat Yourself)
- ✅ Lógica de selección de heads centralizada
- ✅ Métodos helper reutilizables

### Single Responsibility Principle (SRP)
- ✅ `_get_scoring_head()` - Solo obtener scoring head
- ✅ `_compute_class_scores()` - Solo calcular scores
- ✅ `_get_box_head()` - Solo obtener box head
- ✅ `_should_postprocess_multimask()` - Solo verificar condición
- ✅ `_save_multimask_outputs()` - Solo guardar outputs
- ✅ `_select_best_mask()` - Solo seleccionar mejor mask
- ✅ `_apply_best_mask_selection()` - Solo aplicar selección

### KISS (Keep It Simple, Stupid)
- ✅ Métodos pequeños y claros
- ✅ Lógica separada por responsabilidad

### Clean Code
- ✅ Warnings en lugar de print
- ✅ Type hints mejorados
- ✅ Documentación clara

## 🎯 Estado Final

✅ **Métodos Helper Creados**  
✅ **Lógica Centralizada**  
✅ **Warnings Implementados**  
✅ **Código Más Limpio y Mantenible**  
✅ **Mejor Testabilidad**  

## 📝 Archivos Modificados

1. **`sam3_image.py`**
   - ✅ Método helper: `_get_scoring_head()`
   - ✅ Método helper: `_compute_class_scores()`
   - ✅ Método helper: `_get_box_head()`
   - ✅ Método helper: `_should_postprocess_multimask()`
   - ✅ Método helper: `_save_multimask_outputs()`
   - ✅ Método helper: `_select_best_mask()`
   - ✅ Método helper: `_apply_best_mask_selection()`
   - ✅ Reemplazo de print por warnings
   - ✅ Simplificación de `_update_scores_and_boxes()`
   - ✅ Simplificación de `_postprocess_out()`

## ✨ Conclusión

El código está ahora más modularizado y mantenible, siguiendo los mismos principios aplicados en otros módulos. Los métodos helper mejoran la legibilidad y testabilidad del código.

**Refactorización completa.** 🎉

