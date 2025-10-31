from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
import sys
import logging
from typing import List, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from production_code import MultiGPUTrainer, TrainingConfiguration, RadioIntegration
                import time
                import time
            import gradio as gr
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Radio Integration Demo
=====================

This demo showcases the comprehensive radio integration features including:
- Radio station search and playback
- Background music during training
- Audio analysis and feature extraction
- Playlist management
- Gradio interface for radio control
"""


# Add the current directory to the path to import production_code
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RadioIntegrationDemo:
    """Demo class for showcasing radio integration features"""
    
    def __init__(self) -> Any:
        # Configure radio integration
        self.config = TrainingConfiguration(
            enable_radio_integration=True,
            radio_volume=0.5,
            radio_auto_play=False,  # We'll control manually in demo
            radio_quality="high",
            radio_buffer_size=2048,
            radio_sample_rate=44100,
            radio_channels=2
        )
        
        # Initialize trainer with radio integration
        self.trainer = MultiGPUTrainer(self.config)
        self.radio = self.trainer.radio
        
        logger.info("Radio Integration Demo initialized")
    
    def demo_station_search(self) -> Any:
        """Demo radio station search functionality"""
        print("\n" + "="*50)
        print("RADIO STATION SEARCH DEMO")
        print("="*50)
        
        # Search for different types of stations
        search_queries = [
            ("jazz", "US"),
            ("classical", "Germany"),
            ("rock", "UK"),
            ("ambient", None),
            ("news", "US")
        ]
        
        for query, country in search_queries:
            print(f"\nSearching for '{query}' stations" + (f" in {country}" if country else ""))
            stations = self.radio.search_radio_stations(query, country, limit=3)
            
            if stations:
                for i, station in enumerate(stations, 1):
                    print(f"  {i}. {station.get('name', 'Unknown')}")
                    print(f"     URL: {station.get('url', 'N/A')}")
                    print(f"     Type: {station.get('type', 'radio')}")
                    if 'country' in station:
                        print(f"     Country: {station['country']}")
            else:
                print("  No stations found")
    
    def demo_popular_stations(self) -> Any:
        """Demo popular stations functionality"""
        print("\n" + "="*50)
        print("POPULAR RADIO STATIONS DEMO")
        print("="*50)
        
        countries = ["US", "Germany", "UK", None]
        
        for country in countries:
            print(f"\nPopular stations" + (f" in {country}" if country else " worldwide"))
            stations = self.radio.get_popular_stations(country, limit=5)
            
            if stations:
                for i, station in enumerate(stations, 1):
                    print(f"  {i}. {station.get('name', 'Unknown')}")
                    print(f"     Votes: {station.get('votes', 0)}")
                    print(f"     Country: {station.get('country', 'Unknown')}")
            else:
                print("  No popular stations found")
    
    def demo_playback_control(self) -> Any:
        """Demo playback control functionality"""
        print("\n" + "="*50)
        print("PLAYBACK CONTROL DEMO")
        print("="*50)
        
        # Search for a station to play
        print("Searching for ambient music station...")
        stations = self.radio.search_radio_stations("ambient", limit=1)
        
        if stations:
            station = stations[0]
            print(f"Found station: {station['name']}")
            
            # Play the station
            print("Starting playback...")
            success = self.radio.play_station(station['url'], volume=0.3)
            
            if success:
                print("✓ Playback started successfully")
                
                # Get current status
                status = self.trainer.get_radio_status()
                print(f"Radio status: {status}")
                
                # Change volume
                print("Changing volume to 0.7...")
                self.radio.set_volume(0.7)
                
                # Get track info
                track_info = self.radio.get_current_track_info()
                print(f"Track info: {track_info}")
                
                # Stop playback after a few seconds
                print("Playing for 5 seconds...")
                time.sleep(5)
                
                self.radio.stop_playback()
                print("✓ Playback stopped")
            else:
                print("✗ Failed to start playback")
        else:
            print("No stations found for playback demo")
    
    def demo_playlist_management(self) -> List[Any]:
        """Demo playlist management functionality"""
        print("\n" + "="*50)
        print("PLAYLIST MANAGEMENT DEMO")
        print("="*50)
        
        # Create a sample playlist
        playlist_name = "Demo Playlist"
        tracks = [
            "https://stream.example.com/jazz1",
            "https://stream.example.com/classical1",
            "https://stream.example.com/ambient1"
        ]
        
        print(f"Creating playlist: {playlist_name}")
        playlist_id = self.radio.create_playlist(playlist_name, tracks)
        
        if playlist_id:
            print(f"✓ Playlist created with ID: {playlist_id}")
            
            # Load the playlist
            loaded_playlist = self.radio.load_playlist(playlist_id)
            if loaded_playlist:
                print(f"✓ Playlist loaded: {loaded_playlist['name']}")
                print(f"  Tracks: {len(loaded_playlist['tracks'])}")
                print(f"  Created: {loaded_playlist['created_at']}")
            else:
                print("✗ Failed to load playlist")
        else:
            print("✗ Failed to create playlist")
    
    def demo_audio_analysis(self) -> Any:
        """Demo audio analysis functionality"""
        print("\n" + "="*50)
        print("AUDIO ANALYSIS DEMO")
        print("="*50)
        
        # Create sample audio data (sine wave)
        sample_rate = 44100
        duration = 1.0  # 1 second
        frequency = 440.0  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Convert to bytes
        audio_bytes = audio_data.astype(np.float32).tobytes()
        
        # Analyze audio
        print("Analyzing sample audio data...")
        features = self.radio.get_audio_analysis(audio_bytes)
        
        if features:
            print("✓ Audio analysis completed:")
            for feature, value in features.items():
                if isinstance(value, list):
                    print(f"  {feature}: {len(value)} coefficients")
                else:
                    print(f"  {feature}: {value:.4f}")
        else:
            print("✗ Audio analysis failed")
    
    def demo_background_music_training(self) -> Any:
        """Demo background music during training simulation"""
        print("\n" + "="*50)
        print("BACKGROUND MUSIC TRAINING DEMO")
        print("="*50)
        
        # Start background music
        print("Starting background music for training...")
        success = self.trainer.play_background_music("ambient", volume=0.2)
        
        if success:
            print("✓ Background music started")
            
            # Simulate training epochs
            for epoch in range(3):
                print(f"\nTraining epoch {epoch + 1}/3...")
                
                # Get radio status
                status = self.trainer.get_radio_status()
                print(f"Radio status: {status['is_playing']}")
                
                # Simulate training time
                time.sleep(2)
            
            # Stop background music
            self.trainer.stop_background_music()
            print("✓ Background music stopped")
        else:
            print("✗ Failed to start background music")
    
    def demo_gradio_interface(self) -> Any:
        """Demo Gradio interface for radio control"""
        print("\n" + "="*50)
        print("GRADIO INTERFACE DEMO")
        print("="*50)
        
        try:
            
            # Create radio interface
            interface = self.radio.create_radio_interface()
            
            if interface:
                print("✓ Gradio interface created successfully")
                print("Interface components:")
                print("  - Search query input")
                print("  - Country selection")
                print("  - Volume slider")
                print("  - Stop playback button")
                print("  - Status display")
                
                # Launch interface
                print("\nLaunching Gradio interface...")
                interface.launch(
                    server_name="0.0.0.0",
                    server_port=7861,
                    share=False,
                    quiet=True
                )
            else:
                print("✗ Failed to create Gradio interface")
                
        except ImportError:
            print("✗ Gradio not available")
        except Exception as e:
            print(f"✗ Error creating interface: {e}")
    
    async def demo_radio_api_integration(self) -> Any:
        """Demo radio API integrations"""
        print("\n" + "="*50)
        print("RADIO API INTEGRATION DEMO")
        print("="*50)
        
        # Check available APIs
        print("Available radio APIs:")
        for api_name, api_instance in self.radio.radio_apis.items():
            status = "✓ Available" if api_instance else "✗ Not available"
            print(f"  {api_name}: {status}")
        
        # Test specific API functionality
        if self.radio.radio_apis['radio_browser']:
            print("\nTesting Radio Browser API...")
            station_info = self.radio.get_station_info("test_station_id")
            if station_info:
                print("✓ Radio Browser API working")
            else:
                print("✗ Radio Browser API test failed")
    
    def run_complete_demo(self) -> Any:
        """Run the complete radio integration demo"""
        print("🎵 RADIO INTEGRATION DEMO")
        print("="*60)
        
        try:
            # Run all demos
            self.demo_station_search()
            self.demo_popular_stations()
            self.demo_playback_control()
            self.demo_playlist_management()
            self.demo_audio_analysis()
            self.demo_background_music_training()
            self.demo_radio_api_integration()
            self.demo_gradio_interface()
            
            print("\n" + "="*60)
            print("✅ RADIO INTEGRATION DEMO COMPLETED SUCCESSFULLY")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            logger.error(f"Demo error: {e}", exc_info=True)


def create_radio_config_examples():
    """Create example radio configurations"""
    print("\n" + "="*50)
    print("RADIO CONFIGURATION EXAMPLES")
    print("="*50)
    
    configs = {
        "Basic Radio": TrainingConfiguration(
            enable_radio_integration=True,
            radio_volume=0.5,
            radio_auto_play=False
        ),
        
        "High Quality Radio": TrainingConfiguration(
            enable_radio_integration=True,
            radio_quality="high",
            radio_sample_rate=48000,
            radio_buffer_size=4096,
            radio_volume=0.7
        ),
        
        "Background Music Training": TrainingConfiguration(
            enable_radio_integration=True,
            radio_auto_play=True,
            radio_volume=0.3,
            radio_quality="medium"
        ),
        
        "Low Latency Radio": TrainingConfiguration(
            enable_radio_integration=True,
            radio_buffer_size=512,
            radio_sample_rate=22050,
            radio_quality="low"
        )
    }
    
    for name, config in configs.items():
        print(f"\n{name}:")
        print(f"  Quality: {config.radio_quality}")
        print(f"  Sample Rate: {config.radio_sample_rate}")
        print(f"  Buffer Size: {config.radio_buffer_size}")
        print(f"  Volume: {config.radio_volume}")
        print(f"  Auto Play: {config.radio_auto_play}")


def main():
    """Main function to run the radio integration demo"""
    print("🎵 Starting Radio Integration Demo...")
    
    # Create configuration examples
    create_radio_config_examples()
    
    # Run the demo
    demo = RadioIntegrationDemo()
    demo.run_complete_demo()


match __name__:
    case "__main__":
    main() 