# Modular Refactoring - Perplexity System

## Overview

The Perplexity query processing system has been refactored into a modular architecture for better organization, maintainability, and testability.

## New Structure

### Core Module: `core/perplexity/`

The system is now organized into focused modules:

```
core/perplexity/
├── __init__.py          # Public API exports
├── types.py             # Data models and enums
├── detector.py          # Query type detection
├── citations.py         # Citation management
├── formatter.py         # Response formatting
├── prompt_builder.py    # LLM prompt construction
├── processor.py         # Main orchestrator
└── validator.py         # Response validation
```

## Module Responsibilities

### `types.py`
- `QueryType` enum - All supported query types
- `SearchResult` dataclass - Search result representation
- `ProcessedQuery` dataclass - Processed query with metadata

### `detector.py`
- `QueryTypeDetector` - Detects query types from patterns
- Pattern matching for 11 query types
- Confidence scoring

### `citations.py`
- `CitationManager` - Handles citation matching and formatting
- Sentence splitting with LaTeX preservation
- Relevance scoring for search results
- Citation normalization by query type

### `formatter.py`
- `ResponseFormatter` - Formats answers per Perplexity guidelines
- LaTeX normalization
- Header removal
- Query-type specific formatting
- References section removal

### `prompt_builder.py`
- `PromptBuilder` - Builds LLM prompts
- Loads SYSTEM_PROMPT.md
- Formats search results for prompts
- Query-type specific instructions

### `processor.py`
- `PerplexityProcessor` - Main orchestrator
- Coordinates all components
- LLM integration
- Validation integration

### `validator.py`
- `PerplexityValidator` - Validates responses
- Rule checking
- Issue reporting
- Query-type aware validation

## Benefits

### 1. Separation of Concerns
Each module has a single, well-defined responsibility:
- Detection logic isolated
- Formatting logic isolated
- Citation logic isolated
- Validation logic isolated

### 2. Testability
Each module can be tested independently:
```python
from core.perplexity.detector import QueryTypeDetector

detector = QueryTypeDetector()
query_type = detector.detect("What is the weather?")
assert query_type == QueryType.WEATHER
```

### 3. Maintainability
- Easy to locate functionality
- Clear module boundaries
- Reduced coupling
- Better code organization

### 4. Extensibility
Easy to add new features:
- New query types: Add to `detector.py`
- New formatting rules: Add to `formatter.py`
- New validation rules: Add to `validator.py`

### 5. Reusability
Components can be used independently:
```python
from core.perplexity.citations import CitationManager

manager = CitationManager()
cited_text = manager.add_citations(text, search_results)
```

## Migration Guide

### Old Import
```python
from core.perplexity_processor import PerplexityProcessor
```

### New Import
```python
from core.perplexity import PerplexityProcessor
```

All public APIs remain the same - only the internal structure changed.

## Backward Compatibility

The refactoring maintains full backward compatibility:
- Same public API
- Same functionality
- Same behavior
- Only internal organization changed

## File Organization

### Before
```
core/
├── perplexity_processor.py  (900+ lines)
└── perplexity_validator.py  (400+ lines)
```

### After
```
core/
└── perplexity/
    ├── __init__.py          (30 lines)
    ├── types.py             (80 lines)
    ├── detector.py          (120 lines)
    ├── citations.py         (150 lines)
    ├── formatter.py         (200 lines)
    ├── prompt_builder.py    (180 lines)
    ├── processor.py         (200 lines)
    └── validator.py         (400 lines)
```

## Code Quality Improvements

1. **Reduced File Size**: Largest file now ~400 lines (was 900+)
2. **Better Organization**: Related code grouped together
3. **Clearer Dependencies**: Explicit imports show relationships
4. **Easier Navigation**: Find code by responsibility
5. **Better Documentation**: Each module has focused docstrings

## Testing Strategy

Each module can be tested independently:

```python
# Test detector
def test_query_type_detection():
    detector = QueryTypeDetector()
    assert detector.detect("weather today") == QueryType.WEATHER

# Test citations
def test_citation_formatting():
    manager = CitationManager()
    result = manager.add_citations("Text here.", [search_result])
    assert "[1]" in result

# Test formatter
def test_latex_normalization():
    formatter = ResponseFormatter()
    result = formatter._normalize_latex("$x^2$")
    assert "\\(x^2\\)" in result
```

## Future Enhancements

The modular structure makes it easy to:
1. Add new query types
2. Improve citation matching (e.g., with embeddings)
3. Add new formatting rules
4. Extend validation rules
5. Support multiple LLM providers
6. Add response caching
7. Implement streaming responses

## Notes

- All existing code using the Perplexity system continues to work
- No breaking changes to public APIs
- Internal refactoring only
- Improved code organization and maintainability




