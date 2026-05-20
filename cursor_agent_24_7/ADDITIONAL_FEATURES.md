# Additional Features - Perplexity System

## Response Validation System

A comprehensive validation system has been added to ensure all generated answers comply with Perplexity formatting rules.

### Features

- **Automatic Validation**: Validates all generated responses
- **Rule Checking**: Checks all formatting rules and restrictions
- **Issue Reporting**: Provides detailed issues with suggestions
- **Query-Type Aware**: Validates based on query type requirements

### Validation Rules Checked

1. **No Leading Header** - Answer doesn't start with ## or ###
2. **No Ending Question** - Answer doesn't end with ?
3. **Citation Format** - No space before citations, separate brackets
4. **Max Citations** - Maximum 3 citations per sentence
5. **No References Section** - No References or Sources section
6. **LaTeX Format** - Uses \( and \[ not $ or $$
7. **No Emojis** - No emoji characters
8. **Forbidden Phrases** - No "It is important to...", etc.
9. **List Formatting** - No single bullet lists
10. **Query-Type Specific** - URL lookup only [1], translation no citations

### Usage

#### Programmatic Validation

```python
from core.perplexity_processor import PerplexityProcessor

processor = PerplexityProcessor(enable_validation=True)

# Generate answer (automatically validated)
answer = await processor.generate_answer(processed, llm_provider)

# Manual validation
is_valid, issues = processor.validate_response(answer, "academic_research")

if not is_valid:
    for issue in issues:
        print(f"{issue.level}: {issue.message}")
        print(f"  Suggestion: {issue.suggestion}")
```

#### API Validation Endpoint

```bash
POST /api/perplexity/validate
{
  "answer": "Your answer text here...",
  "query_type": "academic_research"
}
```

**Response:**
```json
{
  "valid": false,
  "query_type": "academic_research",
  "issue_count": 2,
  "issues": [
    {
      "level": "error",
      "rule": "citation_no_space",
      "message": "Citation has space before bracket",
      "location": "Position 123",
      "suggestion": "Remove space: 'water12' not 'water 12'"
    }
  ]
}
```

### Validation Levels

- **ERROR**: Violates critical formatting rules (must be fixed)
- **WARNING**: Violates best practices (should be fixed)
- **INFO**: Suggestions for improvement

### Integration

Validation is automatically enabled by default. To disable:

```python
processor = PerplexityProcessor(enable_validation=False)
```

## Enhanced Error Handling

All validation issues are logged but don't prevent answer generation. This allows:

- **Graceful Degradation**: Answers still generated even with minor issues
- **Logging**: All issues logged for monitoring
- **Debugging**: Easy identification of formatting problems

## Example Validation Output

```python
# Valid answer
is_valid, issues = processor.validate_response(
    "Quantum computing uses qubits[1][2] to perform calculations.",
    "academic_research"
)
# is_valid = True, issues = []

# Invalid answer (space before citation)
is_valid, issues = processor.validate_response(
    "Quantum computing uses qubits [1] [2] to perform calculations.",
    "academic_research"
)
# is_valid = False
# issues = [
#   ValidationIssue(
#     level=ERROR,
#     rule="citation_no_space",
#     message="Citation has space before bracket",
#     suggestion="Remove space: 'water12' not 'water 12'"
#   )
# ]
```

## Benefits

1. **Quality Assurance**: Ensures all answers meet Perplexity standards
2. **Debugging**: Identifies formatting issues quickly
3. **Consistency**: Enforces consistent formatting across all responses
4. **Documentation**: Clear suggestions for fixing issues
5. **Monitoring**: Track validation issues over time

## Future Enhancements

Potential improvements:

1. **Auto-Fix**: Automatically fix common formatting issues
2. **Custom Rules**: Allow custom validation rules
3. **Batch Validation**: Validate multiple answers at once
4. **Statistics**: Track validation metrics and trends
5. **Integration Tests**: Automated tests using validator




