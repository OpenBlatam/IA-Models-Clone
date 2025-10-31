# Radio Integration Guide

## Overview

The Radio Integration system provides comprehensive audio streaming capabilities integrated directly into your production training environment. This system supports multiple radio services, background music during training, audio analysis, and playlist management.

## Features

### 🎵 Core Features
- **Multi-Service Radio Support**: Radio Browser, Spotify, Last.fm, Icecast, Shoutcast
- **Background Music**: Automatic music during training sessions
- **Audio Analysis**: Real-time feature extraction and analysis
- **Playlist Management**: Create, save, and load custom playlists
- **Gradio Interface**: Web-based radio control interface
- **Volume Control**: Dynamic volume adjustment
- **Quality Settings**: Configurable audio quality and buffer settings

### 🔧 Technical Features
- **Real-time Streaming**: Low-latency audio streaming
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Memory Efficient**: Optimized buffer management
- **Error Handling**: Robust error recovery and logging
- **API Integration**: Multiple music service APIs
- **Audio Processing**: Advanced audio feature extraction

## Installation

### Prerequisites
```bash
# Install system audio dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio

# Install system audio dependencies (macOS)
brew install portaudio

# Install system audio dependencies (Windows)
# PyAudio should work with pip install
```

### Python Dependencies
```bash
# Install radio integration requirements
pip install -r requirements_radio.txt

# Or install individually
pip install pyaudio librosa soundfile spotipy pylast gradio
```

## Configuration

### Basic Configuration
```python
from production_code import TrainingConfiguration

# Basic radio configuration
config = TrainingConfiguration(
    enable_radio_integration=True,
    radio_volume=0.7,
    radio_auto_play=False,
    radio_quality="high",
    radio_buffer_size=1024,
    radio_sample_rate=44100,
    radio_channels=2
)
```

### Advanced Configuration
```python
# High-quality radio configuration
config = TrainingConfiguration(
    enable_radio_integration=True,
    radio_quality="high",
    radio_sample_rate=48000,
    radio_buffer_size=4096,
    radio_volume=0.8,
    radio_auto_play=True,
    radio_api_key="your_spotify_api_key"  # Optional
)
```

## Usage Examples

### Basic Radio Usage
```python
from production_code import MultiGPUTrainer, TrainingConfiguration

# Initialize trainer with radio
config = TrainingConfiguration(enable_radio_integration=True)
trainer = MultiGPUTrainer(config)

# Search for radio stations
stations = trainer.radio.search_radio_stations("jazz", country="US", limit=5)

# Play a station
if stations:
    success = trainer.radio.play_station(stations[0]['url'], volume=0.5)
    print(f"Playing: {stations[0]['name']}")

# Stop playback
trainer.radio.stop_playback()
```

### Background Music During Training
```python
# Start background music
trainer.play_background_music("ambient", volume=0.3)

# Train with background music
training_history = trainer.train(model, train_dataset, val_dataset, criterion)

# Music automatically stops when training completes
```

### Radio Station Search
```python
# Search by genre and country
jazz_stations = trainer.radio.search_radio_stations("jazz", country="US", limit=10)
classical_stations = trainer.radio.search_radio_stations("classical", country="Germany")

# Search popular stations
popular_stations = trainer.radio.get_popular_stations(country="US", limit=20)

# Get station information
station_info = trainer.radio.get_station_info("station_id")
```

### Playlist Management
```python
# Create a playlist
tracks = [
    "https://stream.example.com/jazz1",
    "https://stream.example.com/classical1",
    "https://stream.example.com/ambient1"
]
playlist_id = trainer.radio.create_playlist("My Training Playlist", tracks)

# Load a playlist
playlist = trainer.radio.load_playlist(playlist_id)
print(f"Playlist: {playlist['name']} with {len(playlist['tracks'])} tracks")
```

### Audio Analysis
```python
# Analyze audio data
audio_data = b"..."  # Your audio bytes
features = trainer.radio.get_audio_analysis(audio_data)

print("Audio Features:")
for feature, value in features.items():
    if isinstance(value, list):
        print(f"  {feature}: {len(value)} coefficients")
    else:
        print(f"  {feature}: {value:.4f}")
```

### Gradio Interface
```python
# Create and launch radio interface
interface = trainer.radio.create_radio_interface()
interface.launch(server_port=7861, share=False)
```

## API Reference

### RadioIntegration Class

#### Core Methods
- `search_radio_stations(query, country=None, language=None, limit=20)`: Search for radio stations
- `get_popular_stations(country=None, limit=20)`: Get popular stations
- `play_station(station_url, volume=None)`: Play a radio station
- `stop_playback()`: Stop current playback
- `set_volume(volume)`: Set playback volume (0.0 to 1.0)

#### Playlist Methods
- `create_playlist(name, tracks)`: Create a new playlist
- `load_playlist(playlist_id)`: Load a saved playlist
- `_save_playlist(playlist_id, playlist)`: Save playlist to storage

#### Analysis Methods
- `get_audio_analysis(audio_data)`: Analyze audio features
- `get_current_track_info()`: Get current track information

#### Interface Methods
- `create_radio_interface()`: Create Gradio web interface

### MultiGPUTrainer Radio Methods
- `play_background_music(station_query, volume)`: Start background music
- `stop_background_music()`: Stop background music
- `get_radio_status()`: Get current radio status
- `search_and_play_radio(query, country, volume)`: Search and play radio
- `create_radio_playlist(name, tracks)`: Create radio playlist
- `get_popular_radio_stations(country, limit)`: Get popular stations

## Configuration Options

### Radio Configuration Parameters
```python
@dataclass
class TrainingConfiguration:
    # Radio integration
    enable_radio_integration: bool = True
    radio_api_key: str = ""
    radio_station_id: str = ""
    radio_playlist_id: str = ""
    radio_volume: float = 0.7
    radio_auto_play: bool = False
    radio_quality: str = "high"  # "low", "medium", "high"
    radio_buffer_size: int = 1024
    radio_sample_rate: int = 44100
    radio_channels: int = 2
```

### Quality Settings
- **Low Quality**: 22050 Hz, 512 buffer, fast streaming
- **Medium Quality**: 44100 Hz, 1024 buffer, balanced
- **High Quality**: 48000 Hz, 4096 buffer, best audio

## Supported Radio Services

### Radio Browser API
- **URL**: https://de1.api.radio-browser.info/
- **Features**: Free, comprehensive station database
- **Usage**: Automatic integration, no API key required

### Spotify API
- **Features**: Premium music streaming
- **Setup**: Requires Spotify API credentials
- **Usage**: Search and play Spotify tracks

### Last.fm API
- **Features**: Music recommendation and scrobbling
- **Setup**: Requires Last.fm API key
- **Usage**: Music discovery and recommendations

### Icecast/Shoutcast
- **Features**: Traditional internet radio
- **Usage**: Direct stream URL playback

## Audio Analysis Features

### Extracted Features
- **RMS Energy**: Root mean square energy
- **Spectral Centroid**: Brightness of sound
- **Spectral Bandwidth**: Frequency spread
- **Spectral Rolloff**: Frequency cutoff point
- **Zero Crossing Rate**: Temporal complexity
- **MFCC**: Mel-frequency cepstral coefficients

### Example Analysis Output
```python
{
    'rms_energy': 0.1234,
    'spectral_centroid': 2345.67,
    'spectral_bandwidth': 1234.56,
    'spectral_rolloff': 3456.78,
    'zero_crossing_rate': 0.0456,
    'mfcc': [1.23, 2.34, 3.45, ...]  # 13 coefficients
}
```

## Error Handling

### Common Issues and Solutions

#### PyAudio Installation Issues
```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio

# macOS
brew install portaudio
pip install pyaudio

# Windows
pip install pyaudio
```

#### Audio Device Issues
```python
# Check available audio devices
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(f"Device {i}: {p.get_device_info_by_index(i)['name']}")
```

#### Network Streaming Issues
```python
# Use lower quality for better streaming
config = TrainingConfiguration(
    radio_quality="low",
    radio_buffer_size=512
)
```

## Performance Optimization

### Memory Management
- Use appropriate buffer sizes for your system
- Monitor memory usage during long streaming sessions
- Clean up audio streams properly

### Network Optimization
- Use lower quality settings for slow connections
- Implement connection retry logic
- Cache frequently accessed station data

### CPU Usage
- Audio analysis can be CPU intensive
- Use lower sample rates for real-time processing
- Consider background processing for analysis

## Integration with Training

### Automatic Background Music
```python
# Enable auto-play during training
config = TrainingConfiguration(
    radio_auto_play=True,
    radio_volume=0.3
)

# Music starts automatically when training begins
trainer.train(model, train_dataset, val_dataset, criterion)
```

### Training Progress Integration
```python
# Monitor radio status during training
for epoch in range(num_epochs):
    # Training code...
    
    # Check radio status
    radio_status = trainer.get_radio_status()
    if radio_status['is_playing']:
        print(f"Background music: {radio_status['current_station']}")
```

## Troubleshooting

### Common Problems

1. **No Audio Output**
   - Check system audio settings
   - Verify PyAudio installation
   - Test with simple audio file

2. **Streaming Issues**
   - Check internet connection
   - Try different radio stations
   - Reduce audio quality settings

3. **API Errors**
   - Verify API keys are correct
   - Check API service status
   - Review rate limits

4. **Memory Issues**
   - Reduce buffer size
   - Stop unused streams
   - Monitor system resources

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
```

## Examples

### Complete Training with Radio
```python
from production_code import MultiGPUTrainer, TrainingConfiguration

# Configure radio integration
config = TrainingConfiguration(
    enable_radio_integration=True,
    radio_auto_play=True,
    radio_volume=0.4,
    radio_quality="medium"
)

# Initialize trainer
trainer = MultiGPUTrainer(config)

# Train with background music
training_history = trainer.train(model, train_dataset, val_dataset, criterion)

# Check final radio status
final_status = trainer.get_radio_status()
print(f"Training completed. Radio status: {final_status}")
```

### Custom Radio Interface
```python
# Create custom radio control
def custom_radio_control():
    # Search for stations
    stations = trainer.radio.search_radio_stations("classical", limit=5)
    
    # Create playlist
    playlist_tracks = [s['url'] for s in stations]
    playlist_id = trainer.radio.create_playlist("Classical Training", playlist_tracks)
    
    # Start background music
    trainer.play_background_music("classical", volume=0.3)
    
    return f"Playing classical music playlist: {playlist_id}"
```

## Future Enhancements

### Planned Features
- **Voice Control**: Voice commands for radio control
- **AI Music Selection**: Intelligent music selection based on training progress
- **Multi-room Audio**: Support for multiple audio outputs
- **Advanced Analytics**: Detailed audio analysis and visualization
- **Cloud Integration**: Cloud-based playlist synchronization

### Contributing
To contribute to the radio integration:
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## License

This radio integration is part of the production training system and follows the same licensing terms.

---

For more information, see the main documentation or run the radio integration demo:
```bash
python radio_integration_demo.py
``` 