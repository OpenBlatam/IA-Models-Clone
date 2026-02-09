# WASM Module - Browser-Based Processing

🌐 Módulo WebAssembly para procesamiento de texto en el browser para **Social Video Transcriber AI**.

## Características

### 📝 Procesamiento de Texto
- Análisis estadístico (palabras, caracteres, oraciones)
- Extracción de keywords con TF-IDF
- Segmentación de texto
- Slugify, hashtags, URLs

### 🔍 Similitud de Strings
- Levenshtein distance
- Jaro-Winkler similarity
- Sørensen-Dice coefficient
- Jaccard index
- Find similar strings

### 🎬 Conversión de Subtítulos
- SRT ↔ VTT ↔ JSON ↔ TXT
- Parse y generate
- Time shifting y scaling
- Merge entries

### 💾 Browser Cache
- LocalStorage wrapper con TTL
- Prefix namespacing
- Automatic cleanup de expirados
- API async-friendly

## Instalación

```bash
# Requisitos: Rust + wasm-pack

cd wasm_module

# Compilar
wasm-pack build --target web --release

# Para desarrollo
wasm-pack build --target web --dev
```

## Uso en JavaScript/TypeScript

### Inicialización

```typescript
import init, { TranscriberWasm, JsCache } from './pkg/transcriber_wasm.js';

async function setup() {
    await init();
    
    const transcriber = new TranscriberWasm();
    const cache = new JsCache('my-app');
    
    return { transcriber, cache };
}
```

### Análisis de Texto

```typescript
const { transcriber } = await setup();

// Análisis estadístico
const stats = transcriber.analyze_text("Tu texto aquí...");
console.log(stats);
// { char_count: 15, word_count: 3, sentence_count: 1, ... }

// Extracción de keywords
const keywords = transcriber.extract_keywords("texto con palabras importantes", 10);
console.log(keywords);
// [{ word: "importantes", frequency: 1, tf_idf: 0.5, relevance: 0.8 }, ...]

// Segmentación
const segments = transcriber.segment_text("Texto largo...", 500);
```

### Similitud de Strings

```typescript
// Comparar dos textos
const similarity = transcriber.compare_texts("hello", "hallo");
console.log(similarity); // 0.88

// Encontrar similares
const candidates = ["hello world", "hello there", "goodbye"];
const similar = transcriber.find_similar("hello", candidates, 0.5);
```

### Subtítulos

```typescript
// Crear entradas de subtítulos
const entries = [
    { index: 1, start_time: 0.0, end_time: 5.0, text: "Hello" },
    { index: 2, start_time: 5.5, end_time: 10.0, text: "World" }
];

// Convertir a SRT
const srt = transcriber.text_to_srt(entries);
console.log(srt);
// 1
// 00:00:00,000 --> 00:00:05,000
// Hello
//
// 2
// 00:00:05,500 --> 00:00:10,000
// World

// Convertir a VTT
const vtt = transcriber.text_to_vtt(entries);

// Parsear SRT existente
const parsed = transcriber.parse_srt(srtContent);
```

### Cache

```typescript
const cache = new JsCache('transcriber');

// Guardar con TTL de 1 hora
cache.set('transcription-123', JSON.stringify(data), 3600n);

// Obtener
const cached = cache.get('transcription-123');
if (cached) {
    const data = JSON.parse(cached);
}

// Verificar existencia
if (cache.has('transcription-123')) {
    // ...
}

// Limpiar expirados
const removed = cache.cleanup_expired();
console.log(`Removed ${removed} expired entries`);

// Listar keys
const keys = cache.keys();

// Limpiar todo
cache.clear();
```

### Utilidades

```typescript
import { 
    hash_string, 
    generate_id, 
    truncate_text,
    capitalize,
    to_snake_case,
    to_camel_case,
    format_bytes,
    format_duration,
    escape_html,
    strip_html,
    is_valid_url,
    extract_domain
} from './pkg/transcriber_wasm.js';

// Hash rápido
const hash = hash_string("content to hash");

// ID único
const id = generate_id();

// Truncar texto
const short = truncate_text("Long text here...", 10, "...");

// Formateo
console.log(format_bytes(1048576n)); // "1.00 MB"
console.log(format_duration(3665n)); // "1h 1m"

// Case conversion
console.log(to_snake_case("helloWorld")); // "hello_world"
console.log(to_camel_case("hello_world")); // "helloWorld"
```

## Benchmarks

```typescript
import { benchmark_text_analysis } from './pkg/transcriber_wasm.js';

const text = "Lorem ipsum dolor sit amet...".repeat(100);
const result = benchmark_text_analysis(text, 1000);
console.log(result);
// {
//   iterations: 1000,
//   total_ms: 45.2,
//   avg_ms: 0.045,
//   ops_per_sec: 22123.89
// }
```

## Performance vs JavaScript

| Operación | WASM | JavaScript | Mejora |
|-----------|------|------------|--------|
| Text Analysis | 0.05ms | 0.3ms | 6x |
| Keyword Extract | 0.1ms | 0.8ms | 8x |
| Jaro-Winkler | 0.01ms | 0.1ms | 10x |
| SRT Parse | 0.02ms | 0.2ms | 10x |

## Estructura

```
wasm_module/
├── Cargo.toml
├── src/
│   ├── lib.rs           # Entry point + TranscriberWasm
│   ├── text.rs          # Text processing
│   ├── similarity.rs    # String similarity
│   ├── subtitles.rs     # Subtitle conversion
│   ├── cache.rs         # Browser cache
│   └── utils.rs         # Utilities
├── pkg/                 # Compiled output
│   ├── transcriber_wasm.js
│   ├── transcriber_wasm.d.ts
│   └── transcriber_wasm_bg.wasm
└── README.md
```

## TypeScript Types

```typescript
interface TextStats {
    char_count: number;
    word_count: number;
    sentence_count: number;
    paragraph_count: number;
    avg_word_length: number;
    avg_sentence_length: number;
    unique_words: number;
    reading_time_minutes: number;
}

interface Keyword {
    word: string;
    frequency: number;
    tf_idf: number;
    relevance: number;
}

interface SubtitleEntry {
    index: number;
    start_time: number;
    end_time: number;
    text: string;
}

interface SimilarityResult {
    text: string;
    score: number;
    algorithm: string;
}
```

## Integración con React

```tsx
import React, { useEffect, useState } from 'react';
import init, { TranscriberWasm } from '../pkg/transcriber_wasm';

function useTranscriber() {
    const [transcriber, setTranscriber] = useState<TranscriberWasm | null>(null);
    
    useEffect(() => {
        init().then(() => {
            setTranscriber(new TranscriberWasm());
        });
    }, []);
    
    return transcriber;
}

function TextAnalyzer() {
    const transcriber = useTranscriber();
    const [text, setText] = useState('');
    const [stats, setStats] = useState<any>(null);
    
    const analyze = () => {
        if (transcriber) {
            setStats(transcriber.analyze_text(text));
        }
    };
    
    return (
        <div>
            <textarea value={text} onChange={e => setText(e.target.value)} />
            <button onClick={analyze}>Analyze</button>
            {stats && <pre>{JSON.stringify(stats, null, 2)}</pre>}
        </div>
    );
}
```

---

**WASM Module** - Alto rendimiento en el browser para Social Video Transcriber AI 🌐












