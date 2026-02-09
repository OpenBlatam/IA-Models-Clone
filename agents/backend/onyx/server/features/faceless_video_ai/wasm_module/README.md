# 🎬 Faceless Video AI - WebAssembly Module

[![WASM](https://img.shields.io/badge/WebAssembly-654FF0?logo=webassembly&logoColor=white)](https://webassembly.org/)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

High-performance browser-based video and image processing using WebAssembly. Process images, preview video effects, and render subtitles entirely client-side.

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Image Processing** | 12+ filters, resize, crop, rotate, flip |
| **Video Preview** | Real-time effects chain processing |
| **Subtitle Rendering** | SRT, VTT, and JSON format support |
| **Color Analysis** | Extract dominant colors from images |
| **Zero Server Load** | All processing happens in the browser |

## 🚀 Performance

Benchmarks on Apple M1 (Safari):

| Operation | 1080p | 4K |
|-----------|-------|-----|
| Grayscale | 2ms | 8ms |
| Blur (r=5) | 15ms | 55ms |
| Resize | 8ms | 25ms |
| Vignette | 5ms | 18ms |
| Full filter chain (5 filters) | 35ms | 120ms |

## 📦 Installation

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install wasm-pack
cargo install wasm-pack
```

### Build

```bash
cd wasm_module

# Development build (fast compile, larger size)
wasm-pack build --target web --dev

# Production build (optimized, ~60% smaller)
wasm-pack build --target web --release

# Node.js target
wasm-pack build --target nodejs --release

# Bundler target (webpack, rollup, etc.)
wasm-pack build --target bundler --release
```

### Output

After building, the `pkg/` directory contains:
- `faceless_video_wasm.js` - JavaScript bindings
- `faceless_video_wasm_bg.wasm` - WebAssembly binary
- `faceless_video_wasm.d.ts` - TypeScript definitions

## 🔧 Usage

### Basic Setup

```typescript
import init, { 
  ImageProcessor, 
  VideoPreview, 
  SubtitleRenderer,
  get_version 
} from 'faceless-video-wasm';

// Initialize WASM (required once)
await init();

console.log(`WASM Version: ${get_version()}`);
```

### Image Processing

```typescript
const processor = new ImageProcessor();

// Load from canvas
processor.load_from_canvas(canvasElement);

// Or load raw data
const imageData = ctx.getImageData(0, 0, width, height);
processor.load_data(imageData.data, width, height);

// Apply filters
processor.apply_filter("brightness", { intensity: 1.2 });
processor.apply_filter("contrast", { intensity: 1.1 });
processor.apply_filter("vignette", { intensity: 0.4 });

// Apply multiple filters at once
processor.apply_filters([
  { name: "grayscale" },
  { name: "brightness", intensity: 1.1 },
  { name: "sharpen", intensity: 0.5 }
]);

// Get result
const result = processor.get_image_data();
ctx.putImageData(result, 0, 0);

// Or draw directly to canvas
processor.draw_to_canvas(outputCanvas);

// Transformations
processor.resize(1920, 1080);
processor.crop(100, 100, 800, 600);
processor.rotate(90);  // 90, 180, or 270
processor.flip_horizontal();
processor.flip_vertical();

// Undo all changes
processor.reset();

// Get dominant colors
const colors = processor.get_dominant_colors(5);
console.log(colors);  // ["#ff0000", "#00ff00", ...]
```

### Video Preview with Effects

```typescript
const preview = new VideoPreview();

// Build effects pipeline
preview.add_effect("brightness", { intensity: 1.1 });
preview.add_effect("contrast", { intensity: 1.2 });
preview.add_effect("vignette", { intensity: 0.3 });

// Process video frame
function processFrame() {
  const imageData = ctx.getImageData(0, 0, width, height);
  const processed = preview.process_frame(imageData, video.currentTime);
  ctx.putImageData(processed, 0, 0);
  requestAnimationFrame(processFrame);
}

// Get active effects
const effects = preview.get_effects();
console.log(effects);  // ["brightness", "contrast", "vignette"]

// Remove specific effect
preview.remove_effect("vignette");

// Clear all effects
preview.clear_effects();
```

### Subtitle Rendering

```typescript
const subtitles = new SubtitleRenderer();

// Load subtitles
subtitles.load_srt(srtContent);  // SRT format
// or
subtitles.load_vtt(vttContent);  // WebVTT format
// or
subtitles.load_json(`[
  {"start": 0, "end": 4, "text": "Hello World"},
  {"start": 5, "end": 10, "text": "Second subtitle"}
]`);

console.log(`Loaded ${subtitles.count} subtitles`);

// Customize appearance
subtitles.set_style(
  32,                          // font size
  "Arial, sans-serif",         // font family
  "#FFFFFF",                   // text color
  "rgba(0, 0, 0, 0.8)"        // background color (or null)
);

subtitles.set_position("bottom");  // "top", "middle", "bottom"
subtitles.set_outline(true);       // black outline for readability

// Render on video frame
function renderFrame() {
  // Draw video frame first
  ctx.drawImage(video, 0, 0);
  
  // Overlay subtitles
  subtitles.render(canvas, video.currentTime);
  
  requestAnimationFrame(renderFrame);
}

// Query subtitles
const currentText = subtitles.get_subtitle_at(5.5);  // Get text at 5.5s
const duration = subtitles.get_duration();           // Total duration
const json = subtitles.to_json();                    // Export as JSON
```

### React Integration

```tsx
import { useEffect, useRef, useState, useCallback } from 'react';
import init, { ImageProcessor } from 'faceless-video-wasm';

function ImageEditor({ imageUrl }: { imageUrl: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [processor, setProcessor] = useState<ImageProcessor | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    init().then(() => {
      setProcessor(new ImageProcessor());
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    if (!canvasRef.current || !processor) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = canvasRef.current!;
      canvas.width = img.width;
      canvas.height = img.height;
      
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0);
      processor.load_from_canvas(canvas);
    };
    img.src = imageUrl;
  }, [imageUrl, processor]);

  const applyFilter = useCallback((filter: string, params = {}) => {
    if (!processor || !canvasRef.current) return;
    
    processor.reset();  // Start fresh
    processor.apply_filter(filter, params);
    processor.draw_to_canvas(canvasRef.current);
  }, [processor]);

  if (loading) return <div>Loading WASM...</div>;

  return (
    <div className="image-editor">
      <canvas ref={canvasRef} />
      <div className="filters">
        <button onClick={() => applyFilter('grayscale')}>Grayscale</button>
        <button onClick={() => applyFilter('sepia')}>Sepia</button>
        <button onClick={() => applyFilter('vintage')}>Vintage</button>
        <button onClick={() => applyFilter('brightness', { intensity: 1.3 })}>
          Bright
        </button>
        <button onClick={() => applyFilter('blur', { radius: 5 })}>Blur</button>
        <button onClick={() => processor?.reset() && processor.draw_to_canvas(canvasRef.current!)}>
          Reset
        </button>
      </div>
    </div>
  );
}
```

## 🎨 Available Filters

| Filter | Parameters | Description |
|--------|------------|-------------|
| `grayscale` | - | Convert to grayscale using luminance |
| `sepia` | - | Warm sepia tone |
| `vintage` | - | Film-like vintage effect |
| `brightness` | `intensity` (0.0-2.0) | Adjust brightness |
| `contrast` | `intensity` (0.0-2.0) | Adjust contrast |
| `saturation` | `intensity` (0.0-2.0) | Adjust color saturation |
| `blur` | `radius` (1-20) | Box blur |
| `sharpen` | `intensity` (0.0-2.0) | Edge sharpening |
| `invert` | - | Invert all colors |
| `vignette` | `intensity` (0.0-1.0) | Darken corners |
| `hue_rotate` | `degrees` (0-360) | Rotate hue |
| `pixelate` | `size` (2-50) | Pixelation effect |
| `noise` | `intensity` (0.0-1.0) | Add film grain |

## 🌐 Browser Support

| Browser | Version | Notes |
|---------|---------|-------|
| Chrome | 89+ | Full support |
| Firefox | 89+ | Full support |
| Safari | 15+ | Full support |
| Edge | 89+ | Full support |

**Requirements:**
- WebAssembly
- Canvas 2D API
- ES2020+ (for async/await)

## 📁 Project Structure

```
wasm_module/
├── Cargo.toml          # Rust dependencies
├── README.md           # This file
├── src/
│   ├── lib.rs          # Main module, ImageProcessor, VideoPreview, SubtitleRenderer
│   └── error.rs        # Error types
└── pkg/                # Build output (generated)
    ├── faceless_video_wasm.js
    ├── faceless_video_wasm_bg.wasm
    └── faceless_video_wasm.d.ts
```

## 🧪 Testing

```bash
# Run WASM tests in headless browser
wasm-pack test --headless --chrome

# Run with Firefox
wasm-pack test --headless --firefox

# Run specific tests
wasm-pack test --headless --chrome -- --filter test_image
```

## 🔧 Development

```bash
# Watch mode (recompile on changes)
cargo watch -s "wasm-pack build --target web --dev"

# Check for issues
cargo clippy --target wasm32-unknown-unknown

# Format code
cargo fmt
```

## 📊 Bundle Size

| Build | Size (gzip) |
|-------|-------------|
| Development | ~1.2 MB |
| Release | ~180 KB |
| Release + wasm-opt | ~140 KB |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `wasm-pack test --headless --chrome`
4. Format: `cargo fmt`
5. Lint: `cargo clippy --target wasm32-unknown-unknown`
6. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.
