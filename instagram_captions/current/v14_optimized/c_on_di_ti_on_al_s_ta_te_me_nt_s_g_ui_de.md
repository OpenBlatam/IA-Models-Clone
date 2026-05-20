# Conditional Statements Guide - Instagram Captions API v14.0

## Overview
This guide documents the proper use of conditional statements in Python, specifically avoiding unnecessary curly braces and following Python best practices.

## ✅ **Correct Python Conditional Statements**

### **1. Simple If Statements**
```python
# ✅ CORRECT - Simple if statement
if config.ENABLE_CACHE:
    self.cache[cache_key] = response.dict()

# ✅ CORRECT - If with else
if response.status == 200:
    successful_requests += 1
else:
    failed_requests += 1

# ✅ CORRECT - If with elif chain
if avg_time < 0.015:
    grade = "ULTRA_FAST"
elif avg_time < 0.025:
    grade = "FAST"
elif avg_time < 0.050:
    grade = "NORMAL"
else:
    grade = "SLOW"
```

### **2. Compound Conditions**
```python
# ✅ CORRECT - Multiple conditions
if config.ENABLE_CACHE and cache_key in self.cache:
    cached_response = self.cache[cache_key]
    self.stats["cache_hits"] += 1

# ✅ CORRECT - Complex conditions
if len(word) > 3 and word.isalpha():
    hashtags.append(f"#{word}")

# ✅ CORRECT - Parentheses for clarity
if (success_rate >= 95 and avg_response_time < 0.025):
    overall_grade = "EXCELLENT"
```

### **3. Conditional Expressions (Ternary)**
```python
# ✅ CORRECT - Ternary operator
torch_dtype = torch.float16 if config.MIXED_PRECISION else torch.float32

# ✅ CORRECT - Conditional assignment
device = "cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu"

# ✅ CORRECT - Conditional return
return caption or self._fallback_generation(request)
```

### **4. Guard Clauses**
```python
# ✅ CORRECT - Early return pattern
if not self.model or not self.tokenizer:
    return self._fallback_generation(request)

# ✅ CORRECT - Validation guard
if not validate_api_key(credentials.credentials):
    raise HTTPException(status_code=401, detail="Invalid API key")

# ✅ CORRECT - Size validation
if len(batch_request.requests) > 100:
    raise HTTPException(status_code=400, detail="Batch size cannot exceed 100")
```

## ❌ **What to Avoid (Other Languages)**

### **1. Unnecessary Curly Braces (C/Java Style)**
```python
# ❌ WRONG - Don't use curly braces in Python
if config.ENABLE_CACHE {
    self.cache[cache_key] = response.dict()
}

# ❌ WRONG - Don't use curly braces with else
if response.status == 200 {
    successful_requests += 1
} else {
    failed_requests += 1
}

# ❌ WRONG - Don't use curly braces in elif chains
if avg_time < 0.015 {
    grade = "ULTRA_FAST"
} elif avg_time < 0.025 {
    grade = "FAST"
} else {
    grade = "SLOW"
}
```

### **2. Unnecessary Parentheses**
```python
# ❌ WRONG - Unnecessary parentheses around simple conditions
if (config.ENABLE_CACHE):
    self.cache[cache_key] = response.dict()

# ❌ WRONG - Unnecessary parentheses around simple comparisons
if (response.status == 200):
    successful_requests += 1

# ✅ CORRECT - Parentheses only when needed for operator precedence
if (a and b) or c:
    pass
```

## 🎯 **Best Practices from v14.0 Implementation**

### **1. Clean Conditional Logic**
```python
# From core/optimized_engine.py
async def _generate_with_ai(self, request: OptimizedRequest) -> str:
    """Generate caption using optimized AI"""
    # Guard clause for early return
    if not self.model or not self.tokenizer:
        return self._fallback_generation(request)
    
    try:
        # ... AI generation logic ...
        
        with torch.no_grad():
            if config.MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    outputs = self.model.generate(...)
            else:
                outputs = self.model.generate(...)
        
        return caption or self._fallback_generation(request)
        
    except Exception as e:
        logger.error(f"AI generation failed: {e}")
        return self._fallback_generation(request)
```

### **2. Performance Grading Logic**
```python
# From routes/performance.py
@router.get("/status", response_model=PerformanceStatusResponse)
async def performance_status() -> PerformanceStatusResponse:
    """Current performance status endpoint"""
    stats = optimized_engine.get_stats()
    perf_summary = performance_monitor.get_performance_summary()
    
    # Clean if-elif chain for grading
    avg_time = perf_summary["avg_response_time"]
    if avg_time < 0.015:
        grade = "ULTRA_FAST"
    elif avg_time < 0.025:
        grade = "FAST"
    elif avg_time < 0.050:
        grade = "NORMAL"
    else:
        grade = "SLOW"
    
    return PerformanceStatusResponse(
        performance_grade=grade,
        average_response_time=avg_time,
        cache_hit_rate=stats["cache_hit_rate"],
        total_requests=stats["total_requests"],
        uptime=perf_summary["uptime"]
    )
```

### **3. Validation Logic**
```python
# From utils/validators.py
def validate_performance_thresholds(
    avg_response_time: float,
    cache_hit_rate: float,
    success_rate: float
) -> dict:
    """Validate performance thresholds - pure function"""
    thresholds = {
        "response_time_grade": "SLOW",
        "cache_grade": "POOR", 
        "success_grade": "POOR"
    }
    
    # Clean if-elif chains for grading
    if avg_response_time < 0.015:
        thresholds["response_time_grade"] = "ULTRA_FAST"
    elif avg_response_time < 0.025:
        thresholds["response_time_grade"] = "FAST"
    elif avg_response_time < 0.050:
        thresholds["response_time_grade"] = "NORMAL"
    
    if cache_hit_rate >= 95:
        thresholds["cache_grade"] = "EXCELLENT"
    elif cache_hit_rate >= 80:
        thresholds["cache_grade"] = "GOOD"
    elif cache_hit_rate >= 60:
        thresholds["cache_grade"] = "FAIR"
    
    if success_rate >= 99:
        thresholds["success_grade"] = "EXCELLENT"
    elif success_rate >= 95:
        thresholds["success_grade"] = "GOOD"
    elif success_rate >= 90:
        thresholds["success_grade"] = "FAIR"
    
    return thresholds
```

## 🔧 **Conditional Statement Patterns**

### **1. Guard Clauses**
```python
# ✅ RECOMMENDED - Early returns for validation
def process_request(request: OptimizedRequest) -> OptimizedResponse:
    if not request.content_description:
        raise ValueError("Content description is required")
    
    if request.hashtag_count < 5:
        request.hashtag_count = 5
    
    if request.hashtag_count > 30:
        request.hashtag_count = 30
    
    # Main logic here
    return generate_caption(request)
```

### **2. Conditional Assignment**
```python
# ✅ RECOMMENDED - Conditional assignment
def get_optimization_level(performance_grade: str) -> str:
    if performance_grade == "ULTRA_FAST":
        return "ultra_fast"
    elif performance_grade == "FAST":
        return "balanced"
    else:
        return "quality"

# ✅ RECOMMENDED - Using ternary for simple cases
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
```

### **3. Nested Conditions**
```python
# ✅ RECOMMENDED - Flatten nested conditions
def process_batch(requests: List[OptimizedRequest]) -> List[OptimizedResponse]:
    if not requests:
        return []
    
    if len(requests) > 100:
        raise ValueError("Batch size too large")
    
    # Process requests
    return [process_single(req) for req in requests]

# ❌ AVOID - Deep nesting
def process_batch_bad(requests: List[OptimizedRequest]) -> List[OptimizedResponse]:
    if requests:
        if len(requests) <= 100:
            return [process_single(req) for req in requests]
        else:
            raise ValueError("Batch size too large")
    else:
        return []
```

## 📊 **Performance Considerations**

### **1. Short-Circuit Evaluation**
```python
# ✅ EFFICIENT - Short-circuit evaluation
if config.ENABLE_CACHE and cache_key in self.cache:
    # cache_key in self.cache only evaluated if ENABLE_CACHE is True
    return cached_response

# ✅ EFFICIENT - Guard clauses prevent unnecessary work
if not self.model or not self.tokenizer:
    return self._fallback_generation(request)
# No need to proceed with AI generation if models aren't loaded
```

### **2. Conditional Compilation**
```python
# ✅ EFFICIENT - Conditional imports and features
if config.USE_GPU:
    import torch.cuda.amp
    # GPU-specific optimizations

if config.ENABLE_JIT:
    # JIT compilation features
    pass
```

## 🧪 **Testing Conditional Logic**

### **1. Unit Tests for Conditions**
```python
def test_performance_grading():
    """Test performance grading logic"""
    # Test ultra-fast condition
    result = validate_performance_thresholds(0.010, 95, 99)
    assert result["response_time_grade"] == "ULTRA_FAST"
    
    # Test fast condition
    result = validate_performance_thresholds(0.020, 95, 99)
    assert result["response_time_grade"] == "FAST"
    
    # Test normal condition
    result = validate_performance_thresholds(0.040, 95, 99)
    assert result["response_time_grade"] == "NORMAL"
    
    # Test slow condition
    result = validate_performance_thresholds(0.100, 95, 99)
    assert result["response_time_grade"] == "SLOW"
```

### **2. Edge Case Testing**
```python
def test_edge_cases():
    """Test edge cases in conditional logic"""
    # Test empty content
    with pytest.raises(ValueError):
        sanitize_content("")
    
    # Test boundary values
    assert validate_hashtag_count(5) == True
    assert validate_hashtag_count(30) == True
    assert validate_hashtag_count(4) == False
    assert validate_hashtag_count(31) == False
```

## 📝 **Code Review Checklist**

### **Conditional Statement Review**
- [ ] No unnecessary curly braces
- [ ] Proper indentation (4 spaces)
- [ ] Guard clauses used where appropriate
- [ ] Short-circuit evaluation leveraged
- [ ] Complex conditions properly parenthesized
- [ ] Nested conditions flattened when possible
- [ ] Edge cases handled
- [ ] Performance implications considered

### **Common Issues to Watch For**
- [ ] Missing colons after if/elif/else
- [ ] Incorrect indentation
- [ ] Unnecessary parentheses
- [ ] Deep nesting
- [ ] Missing edge cases
- [ ] Inefficient condition ordering

## 🎯 **Summary**

The Instagram Captions API v14.0 follows Python best practices for conditional statements:

✅ **Clean and Readable**: No unnecessary curly braces or parentheses
✅ **Efficient**: Leverages short-circuit evaluation and guard clauses
✅ **Maintainable**: Clear logic flow with early returns
✅ **Testable**: Well-structured conditions for comprehensive testing
✅ **Performant**: Optimized condition ordering for speed

This approach ensures the code is:
- **Pythonic**: Follows Python conventions and idioms
- **Efficient**: Optimized for performance
- **Maintainable**: Easy to understand and modify
- **Reliable**: Handles edge cases properly
- **Testable**: Clear logic for comprehensive testing 