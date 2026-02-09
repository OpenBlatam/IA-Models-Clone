# Perplexity-Style Query Processing - Enhanced Implementation

## Overview

The `cursor_agent_24_7` has been enhanced with a complete Perplexity-style query processing system that includes:

- **Query Type Detection** - Automatically detects 11 different query types
- **Citation Management** - Intelligent citation matching and formatting
- **LLM Integration** - Ready-to-use LLM prompt building and answer generation
- **Response Formatting** - Perplexity-style formatting with proper markdown
- **RESTful API** - Complete API endpoints for integration

## Key Enhancements

### 1. Intelligent Citation System

The citation system now uses keyword matching and content similarity to automatically add citations `[1]`, `[2]`, etc. to sentences based on relevant search results.

**Features:**
- Keyword matching between sentences and search results
- Phrase matching for higher accuracy
- Automatic citation formatting (max 3 citations per sentence)
- No manual citation placement needed

### 2. Prompt Builder

A new `PromptBuilder` class that:
- Loads the system prompt from `SYSTEM_PROMPT.md`
- Builds complete LLM prompts with query context
- Includes query-type-specific instructions
- Formats search results for LLM consumption

**Usage:**
```python
from core.perplexity_processor import PerplexityProcessor

processor = PerplexityProcessor()
processed = processor.process_query(query, search_results)
prompt = processor.build_llm_prompt(processed)
```

### 3. LLM Integration

The processor now supports multiple LLM providers:

- **LLMPipeline** (from `core.llm_pipeline`)
- **OpenAI-style clients** (OpenAI, Anthropic, etc.)
- **Custom async callables**
- **Sync providers** (automatically wrapped)

**Usage:**
```python
# With LLM provider
answer = await processor.generate_answer(processed, llm_provider)

# Without LLM (placeholder)
answer = await processor.generate_answer(processed, None)
```

### 4. Enhanced API Endpoints

#### POST `/api/perplexity/query`
Process a query and generate a formatted answer.

**Request:**
```json
{
  "query": "What is quantum computing?",
  "search_results": [
    {
      "title": "Quantum Computing Guide",
      "url": "https://example.com",
      "snippet": "Quantum computing uses qubits...",
      "content": "Full content here..."
    }
  ],
  "use_llm": false,
  "include_metadata": true
}
```

**Response:**
```json
{
  "query": "What is quantum computing?",
  "query_type": "academic_research",
  "answer": "Formatted answer with citations[1][2]...",
  "metadata": {...}
}
```

#### POST `/api/perplexity/prompt`
Get the LLM prompt without generating an answer (useful for debugging).

**Request:**
```json
{
  "query": "What is Python?",
  "search_results": [...]
}
```

**Response:**
```json
{
  "query": "What is Python?",
  "query_type": "coding",
  "prompt": "Complete prompt text...",
  "metadata": {...}
}
```

#### POST `/api/perplexity/process`
Process query and return metadata only (no answer generation).

#### GET `/api/perplexity/query-types`
Get list of all supported query types.

## Query Types Supported

1. **Academic Research** - Scientific write-ups with detailed sections
2. **Recent News** - News summaries grouped by topics
3. **Weather** - Short weather forecasts
4. **People** - Biographical information
5. **Coding** - Code-first responses with explanations
6. **Cooking Recipes** - Step-by-step recipes
7. **Translation** - Language translations (no citations)
8. **Creative Writing** - Creative content (no citations)
9. **Science/Math** - Mathematical calculations
10. **URL Lookup** - URL content summarization
11. **General** - Default for other queries

## Citation Format

Citations are automatically added in the format: `[1][2][3]` at the end of sentences.

**Example:**
```
Quantum computing uses quantum mechanical phenomena to perform computations[1]. 
Unlike classical bits, qubits can exist in superposition[2][3].
```

## Integration Examples

### Basic Usage (No LLM)

```python
from core.perplexity_processor import PerplexityProcessor

processor = PerplexityProcessor()
processed = processor.process_query(
    query="What is the weather today?",
    search_results=[...]
)
answer = await processor.generate_answer(processed, None)
```

### With LLM Provider

```python
from core.perplexity_processor import PerplexityProcessor
from core.llm_pipeline import LLMPipeline, LLMConfig

# Setup LLM
llm_config = LLMConfig(model="gpt-4", temperature=0.7)
llm = LLMPipeline(llm_config)

# Process and generate
processor = PerplexityProcessor()
processed = processor.process_query(query, search_results)
answer = await processor.generate_answer(processed, llm)
```

### With OpenAI Client

```python
import openai
from core.perplexity_processor import PerplexityProcessor

openai_client = openai.AsyncOpenAI(api_key="...")
processor = PerplexityProcessor()
processed = processor.process_query(query, search_results)
answer = await processor.generate_answer(processed, openai_client)
```

## System Prompt Integration

The system automatically loads `SYSTEM_PROMPT.md` and uses it to build LLM prompts. The prompt includes:

- Goal and identity instructions
- Formatting rules
- Query type-specific instructions
- Search results context
- Citation guidelines

## Response Formatting Rules

All responses follow Perplexity guidelines:

- **Start**: Few sentences summary (no header)
- **Sections**: Level 2 headers (##)
- **Lists**: Flat lists, prefer unordered
- **Citations**: `[1][2]` format, max 3 per sentence
- **Code**: Markdown code blocks with language
- **Math**: LaTeX format `\(formula\)`
- **End**: Summary sentences

## Error Handling

The system gracefully handles:

- Missing LLM providers (uses placeholder)
- Missing search results (still processes query)
- Invalid query types (defaults to general)
- Citation errors (continues without citations)

## Performance Considerations

- Citation matching uses efficient keyword matching
- Search results are indexed for fast lookup
- LLM calls are async for better concurrency
- Prompt building is cached when possible

## Future Enhancements

Potential improvements:

1. **NLP-based Citation Matching** - Use embeddings for better citation accuracy
2. **Multi-LLM Support** - Support for multiple LLM providers simultaneously
3. **Streaming Responses** - Stream LLM responses as they're generated
4. **Citation Validation** - Verify citations match actual content
5. **Query Type Learning** - ML-based query type detection
6. **Response Caching** - Cache formatted responses for similar queries

## Testing

Test the endpoints:

```bash
# Start API
python -m agents.backend.onyx.server.features.cursor_agent_24_7.main --mode api

# Test query processing
curl -X POST http://localhost:8024/api/perplexity/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Python?",
    "search_results": [{
      "title": "Python Programming",
      "url": "https://python.org",
      "snippet": "Python is a high-level programming language"
    }]
  }'
```

## Notes

- The system prompt is loaded from `SYSTEM_PROMPT.md` in the feature directory
- LLM integration is optional - works without LLM (uses placeholders)
- Citations are automatically added based on content matching
- All formatting follows Perplexity-style guidelines strictly




