#!/usr/bin/env python3
"""
Space Technology Integration System

Advanced space technology integration with:
- Satellite communication networks
- Space-based data processing
- Orbital mechanics and positioning
- Space weather monitoring
- Satellite constellation management
- Deep space communication protocols
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
import asyncio
import time
import json
import uuid
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from collections import defaultdict, deque
import math
import ephem
from skyfield.api import load, EarthSatellite
from skyfield.timelib import Time

logger = structlog.get_logger("space_technology")

# =============================================================================
# SPACE TECHNOLOGY MODELS
# =============================================================================

class SatelliteType(Enum):
    """Satellite types."""
    LEO = "leo"  # Low Earth Orbit
    MEO = "meo"  # Medium Earth Orbit
    GEO = "geo"  # Geostationary Earth Orbit
    DEEP_SPACE = "deep_space"
    CUBESAT = "cubesat"
    COMMUNICATION = "communication"
    OBSERVATION = "observation"
    NAVIGATION = "navigation"
    WEATHER = "weather"

class OrbitType(Enum):
    """Orbit types."""
    CIRCULAR = "circular"
    ELLIPTICAL = "elliptical"
    POLAR = "polar"
    EQUATORIAL = "equatorial"
    SUN_SYNCHRONOUS = "sun_synchronous"
    MOLNIYA = "molniya"
    TUNDRA = "tundra"

class CommunicationProtocol(Enum):
    """Space communication protocols."""
    CCSDS = "ccsds"
    S_BAND = "s_band"
    X_BAND = "x_band"
    KA_BAND = "ka_band"
    OPTICAL = "optical"
    QUANTUM = "quantum"
    LASER = "laser"

@dataclass
class Satellite:
    """Satellite information."""
    satellite_id: str
    name: str
    satellite_type: SatelliteType
    orbit_type: OrbitType
    altitude: float  # km
    inclination: float  # degrees
    period: float  # minutes
    launch_date: datetime
    status: str
    capabilities: List[str]
    power_generation: float  # watts
    data_capacity: int  # bytes
    communication_protocols: List[CommunicationProtocol]
    position: Dict[str, float]  # lat, lng, alt
    velocity: Dict[str, float]  # km/s
    last_contact: datetime
    mission_life: int  # years
    
    def __post_init__(self):
        if not self.satellite_id:
            self.satellite_id = str(uuid.uuid4())
        if not self.last_contact:
            self.last_contact = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "satellite_id": self.satellite_id,
            "name": self.name,
            "satellite_type": self.satellite_type.value,
            "orbit_type": self.orbit_type.value,
            "altitude": self.altitude,
            "inclination": self.inclination,
            "period": self.period,
            "launch_date": self.launch_date.isoformat(),
            "status": self.status,
            "capabilities": self.capabilities,
            "power_generation": self.power_generation,
            "data_capacity": self.data_capacity,
            "communication_protocols": [p.value for p in self.communication_protocols],
            "position": self.position,
            "velocity": self.velocity,
            "last_contact": self.last_contact.isoformat(),
            "mission_life": self.mission_life
        }

@dataclass
class GroundStation:
    """Ground station information."""
    station_id: str
    name: str
    location: Dict[str, float]  # lat, lng, alt
    antenna_diameter: float  # meters
    frequency_bands: List[str]
    tracking_capability: bool
    data_rate: float  # Mbps
    availability: float  # percentage
    weather_dependency: bool
    last_maintenance: datetime
    status: str
    
    def __post_init__(self):
        if not self.station_id:
            self.station_id = str(uuid.uuid4())
        if not self.last_maintenance:
            self.last_maintenance = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "station_id": self.station_id,
            "name": self.name,
            "location": self.location,
            "antenna_diameter": self.antenna_diameter,
            "frequency_bands": self.frequency_bands,
            "tracking_capability": self.tracking_capability,
            "data_rate": self.data_rate,
            "availability": self.availability,
            "weather_dependency": self.weather_dependency,
            "last_maintenance": self.last_maintenance.isoformat(),
            "status": self.status
        }

@dataclass
class CommunicationLink:
    """Satellite communication link."""
    link_id: str
    satellite_id: str
    ground_station_id: str
    protocol: CommunicationProtocol
    frequency: float  # MHz
    bandwidth: float  # MHz
    data_rate: float  # Mbps
    signal_strength: float  # dB
    link_quality: float  # 0.0 to 1.0
    established_at: datetime
    duration: float  # minutes
    data_transferred: int  # bytes
    
    def __post_init__(self):
        if not self.link_id:
            self.link_id = str(uuid.uuid4())
        if not self.established_at:
            self.established_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "link_id": self.link_id,
            "satellite_id": self.satellite_id,
            "ground_station_id": self.ground_station_id,
            "protocol": self.protocol.value,
            "frequency": self.frequency,
            "bandwidth": self.bandwidth,
            "data_rate": self.data_rate,
            "signal_strength": self.signal_strength,
            "link_quality": self.link_quality,
            "established_at": self.established_at.isoformat(),
            "duration": self.duration,
            "data_transferred": self.data_transferred
        }

@dataclass
class SpaceWeather:
    """Space weather information."""
    weather_id: str
    timestamp: datetime
    solar_flux: float
    geomagnetic_index: float
    proton_flux: float
    electron_flux: float
    radiation_level: float
    aurora_activity: float
    ionospheric_disturbance: float
    impact_on_communications: str
    impact_on_satellites: str
    
    def __post_init__(self):
        if not self.weather_id:
            self.weather_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "weather_id": self.weather_id,
            "timestamp": self.timestamp.isoformat(),
            "solar_flux": self.solar_flux,
            "geomagnetic_index": self.geomagnetic_index,
            "proton_flux": self.proton_flux,
            "electron_flux": self.electron_flux,
            "radiation_level": self.radiation_level,
            "aurora_activity": self.aurora_activity,
            "ionospheric_disturbance": self.ionospheric_disturbance,
            "impact_on_communications": self.impact_on_communications,
            "impact_on_satellites": self.impact_on_satellites
        }

# =============================================================================
# SPACE TECHNOLOGY MANAGER
# =============================================================================

class SpaceTechnologyManager:
    """Space technology management system."""
    
    def __init__(self):
        self.satellites: Dict[str, Satellite] = {}
        self.ground_stations: Dict[str, GroundStation] = {}
        self.communication_links: Dict[str, CommunicationLink] = {}
        self.space_weather: deque = deque(maxlen=1000)
        
        # Orbital mechanics
        self.ephemeris = load('de421.bsp')
        self.earth = self.ephemeris['earth']
        
        # Statistics
        self.stats = {
            'total_satellites': 0,
            'active_satellites': 0,
            'total_ground_stations': 0,
            'active_ground_stations': 0,
            'total_communication_links': 0,
            'active_communication_links': 0,
            'total_data_transferred': 0,
            'average_link_quality': 0.0,
            'space_weather_alerts': 0
        }
        
        # Background tasks
        self.orbital_tracking_task: Optional[asyncio.Task] = None
        self.space_weather_task: Optional[asyncio.Task] = None
        self.communication_monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def start(self) -> None:
        """Start the space technology manager."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Initialize default satellites and ground stations
        await self._initialize_default_assets()
        
        # Start background tasks
        self.orbital_tracking_task = asyncio.create_task(self._orbital_tracking_loop())
        self.space_weather_task = asyncio.create_task(self._space_weather_loop())
        self.communication_monitoring_task = asyncio.create_task(self._communication_monitoring_loop())
        
        logger.info("Space Technology Manager started")
    
    async def stop(self) -> None:
        """Stop the space technology manager."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        if self.orbital_tracking_task:
            self.orbital_tracking_task.cancel()
        if self.space_weather_task:
            self.space_weather_task.cancel()
        if self.communication_monitoring_task:
            self.communication_monitoring_task.cancel()
        
        logger.info("Space Technology Manager stopped")
    
    async def _initialize_default_assets(self) -> None:
        """Initialize default satellites and ground stations."""
        # Initialize LEO communication satellites
        leo_satellites = [
            Satellite(
                name="Video-OpusClip-LEO-001",
                satellite_type=SatelliteType.LEO,
                orbit_type=OrbitType.CIRCULAR,
                altitude=550,
                inclination=53.0,
                period=95.0,
                launch_date=datetime(2024, 1, 1),
                status="active",
                capabilities=["video_streaming", "data_relay", "real_time_communication"],
                power_generation=2000,
                data_capacity=1000000000,  # 1 GB
                communication_protocols=[CommunicationProtocol.S_BAND, CommunicationProtocol.X_BAND],
                position={"lat": 0, "lng": 0, "alt": 550},
                velocity={"x": 7.5, "y": 0, "z": 0},
                mission_life=5
            ),
            Satellite(
                name="Video-OpusClip-LEO-002",
                satellite_type=SatelliteType.LEO,
                orbit_type=OrbitType.CIRCULAR,
                altitude=550,
                inclination=53.0,
                period=95.0,
                launch_date=datetime(2024, 1, 15),
                status="active",
                capabilities=["video_streaming", "data_relay", "real_time_communication"],
                power_generation=2000,
                data_capacity=1000000000,
                communication_protocols=[CommunicationProtocol.S_BAND, CommunicationProtocol.X_BAND],
                position={"lat": 0, "lng": 0, "alt": 550},
                velocity={"x": 7.5, "y": 0, "z": 0},
                mission_life=5
            )
        ]
        
        for satellite in leo_satellites:
            self.satellites[satellite.satellite_id] = satellite
        
        # Initialize GEO communication satellite
        geo_satellite = Satellite(
            name="Video-OpusClip-GEO-001",
            satellite_type=SatelliteType.GEO,
            orbit_type=OrbitType.EQUATORIAL,
            altitude=35786,
            inclination=0.0,
            period=1440.0,
            launch_date=datetime(2023, 12, 1),
            status="active",
            capabilities=["video_streaming", "data_relay", "broadcast"],
            power_generation=5000,
            data_capacity=10000000000,  # 10 GB
            communication_protocols=[CommunicationProtocol.KA_BAND, CommunicationProtocol.X_BAND],
            position={"lat": 0, "lng": -75, "alt": 35786},
            velocity={"x": 0, "y": 0, "z": 0},
            mission_life=15
        )
        
        self.satellites[geo_satellite.satellite_id] = geo_satellite
        
        # Initialize ground stations
        ground_stations = [
            GroundStation(
                name="Video-OpusClip-GS-001",
                location={"lat": 40.7128, "lng": -74.0060, "alt": 10},  # New York
                antenna_diameter=13.0,
                frequency_bands=["S-Band", "X-Band", "Ka-Band"],
                tracking_capability=True,
                data_rate=1000.0,
                availability=99.5,
                weather_dependency=True,
                status="active"
            ),
            GroundStation(
                name="Video-OpusClip-GS-002",
                location={"lat": 51.5074, "lng": -0.1278, "alt": 20},  # London
                antenna_diameter=11.0,
                frequency_bands=["S-Band", "X-Band"],
                tracking_capability=True,
                data_rate=800.0,
                availability=99.0,
                weather_dependency=True,
                status="active"
            ),
            GroundStation(
                name="Video-OpusClip-GS-003",
                location={"lat": 35.6762, "lng": 139.6503, "alt": 15},  # Tokyo
                antenna_diameter=12.0,
                frequency_bands=["S-Band", "X-Band", "Ka-Band"],
                tracking_capability=True,
                data_rate=1200.0,
                availability=99.8,
                weather_dependency=True,
                status="active"
            )
        ]
        
        for station in ground_stations:
            self.ground_stations[station.station_id] = station
        
        # Update statistics
        self.stats['total_satellites'] = len(self.satellites)
        self.stats['active_satellites'] = len([s for s in self.satellites.values() if s.status == "active"])
        self.stats['total_ground_stations'] = len(self.ground_stations)
        self.stats['active_ground_stations'] = len([s for s in self.ground_stations.values() if s.status == "active"])
    
    def add_satellite(self, satellite: Satellite) -> str:
        """Add a satellite to the constellation."""
        self.satellites[satellite.satellite_id] = satellite
        self.stats['total_satellites'] += 1
        if satellite.status == "active":
            self.stats['active_satellites'] += 1
        
        logger.info(
            "Satellite added",
            satellite_id=satellite.satellite_id,
            name=satellite.name,
            type=satellite.satellite_type.value
        )
        
        return satellite.satellite_id
    
    def add_ground_station(self, station: GroundStation) -> str:
        """Add a ground station."""
        self.ground_stations[station.station_id] = station
        self.stats['total_ground_stations'] += 1
        if station.status == "active":
            self.stats['active_ground_stations'] += 1
        
        logger.info(
            "Ground station added",
            station_id=station.station_id,
            name=station.name
        )
        
        return station.station_id
    
    async def establish_communication_link(self, satellite_id: str, ground_station_id: str,
                                         protocol: CommunicationProtocol) -> str:
        """Establish communication link between satellite and ground station."""
        if satellite_id not in self.satellites or ground_station_id not in self.ground_stations:
            raise ValueError("Satellite or ground station not found")
        
        satellite = self.satellites[satellite_id]
        ground_station = self.ground_stations[ground_station_id]
        
        # Calculate link parameters
        distance = self._calculate_distance(satellite.position, ground_station.location)
        signal_strength = self._calculate_signal_strength(distance, protocol)
        link_quality = self._calculate_link_quality(signal_strength, protocol)
        
        # Create communication link
        link = CommunicationLink(
            satellite_id=satellite_id,
            ground_station_id=ground_station_id,
            protocol=protocol,
            frequency=self._get_frequency_for_protocol(protocol),
            bandwidth=self._get_bandwidth_for_protocol(protocol),
            data_rate=self._get_data_rate_for_protocol(protocol),
            signal_strength=signal_strength,
            link_quality=link_quality,
            duration=0.0,
            data_transferred=0
        )
        
        self.communication_links[link.link_id] = link
        self.stats['total_communication_links'] += 1
        self.stats['active_communication_links'] += 1
        
        logger.info(
            "Communication link established",
            link_id=link.link_id,
            satellite_id=satellite_id,
            ground_station_id=ground_station_id,
            protocol=protocol.value,
            signal_strength=signal_strength,
            link_quality=link_quality
        )
        
        return link.link_id
    
    def _calculate_distance(self, pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
        """Calculate distance between two positions."""
        # Convert to Cartesian coordinates (simplified)
        x1, y1, z1 = pos1.get('lng', 0), pos1.get('lat', 0), pos1.get('alt', 0)
        x2, y2, z2 = pos2.get('lng', 0), pos2.get('lat', 0), pos2.get('alt', 0)
        
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    def _calculate_signal_strength(self, distance: float, protocol: CommunicationProtocol) -> float:
        """Calculate signal strength based on distance and protocol."""
        # Simplified free space path loss calculation
        frequency = self._get_frequency_for_protocol(protocol)
        path_loss = 20 * math.log10(distance) + 20 * math.log10(frequency) + 32.45
        
        # Base signal strength (dBm)
        base_power = {
            CommunicationProtocol.S_BAND: 40,
            CommunicationProtocol.X_BAND: 35,
            CommunicationProtocol.KA_BAND: 30,
            CommunicationProtocol.OPTICAL: 20,
            CommunicationProtocol.LASER: 15
        }
        
        return base_power.get(protocol, 30) - path_loss
    
    def _calculate_link_quality(self, signal_strength: float, protocol: CommunicationProtocol) -> float:
        """Calculate link quality based on signal strength."""
        # Signal strength thresholds (dBm)
        thresholds = {
            CommunicationProtocol.S_BAND: (-80, -60),
            CommunicationProtocol.X_BAND: (-85, -65),
            CommunicationProtocol.KA_BAND: (-90, -70),
            CommunicationProtocol.OPTICAL: (-95, -75),
            CommunicationProtocol.LASER: (-100, -80)
        }
        
        min_threshold, max_threshold = thresholds.get(protocol, (-80, -60))
        
        if signal_strength >= max_threshold:
            return 1.0
        elif signal_strength <= min_threshold:
            return 0.0
        else:
            return (signal_strength - min_threshold) / (max_threshold - min_threshold)
    
    def _get_frequency_for_protocol(self, protocol: CommunicationProtocol) -> float:
        """Get frequency for communication protocol."""
        frequencies = {
            CommunicationProtocol.S_BAND: 2400,  # MHz
            CommunicationProtocol.X_BAND: 8000,  # MHz
            CommunicationProtocol.KA_BAND: 20000,  # MHz
            CommunicationProtocol.OPTICAL: 300000,  # MHz
            CommunicationProtocol.LASER: 300000  # MHz
        }
        return frequencies.get(protocol, 2400)
    
    def _get_bandwidth_for_protocol(self, protocol: CommunicationProtocol) -> float:
        """Get bandwidth for communication protocol."""
        bandwidths = {
            CommunicationProtocol.S_BAND: 10,  # MHz
            CommunicationProtocol.X_BAND: 50,  # MHz
            CommunicationProtocol.KA_BAND: 100,  # MHz
            CommunicationProtocol.OPTICAL: 1000,  # MHz
            CommunicationProtocol.LASER: 1000  # MHz
        }
        return bandwidths.get(protocol, 10)
    
    def _get_data_rate_for_protocol(self, protocol: CommunicationProtocol) -> float:
        """Get data rate for communication protocol."""
        data_rates = {
            CommunicationProtocol.S_BAND: 10,  # Mbps
            CommunicationProtocol.X_BAND: 100,  # Mbps
            CommunicationProtocol.KA_BAND: 500,  # Mbps
            CommunicationProtocol.OPTICAL: 1000,  # Mbps
            CommunicationProtocol.LASER: 2000  # Mbps
        }
        return data_rates.get(protocol, 10)
    
    async def _orbital_tracking_loop(self) -> None:
        """Orbital tracking loop."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                for satellite in self.satellites.values():
                    if satellite.status != "active":
                        continue
                    
                    # Update satellite position (simplified orbital mechanics)
                    self._update_satellite_position(satellite, current_time)
                    
                    # Update last contact
                    satellite.last_contact = current_time
                
                await asyncio.sleep(60)  # Update every minute
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Orbital tracking error", error=str(e))
                await asyncio.sleep(60)
    
    def _update_satellite_position(self, satellite: Satellite, current_time: datetime) -> None:
        """Update satellite position using orbital mechanics."""
        # Simplified orbital mechanics calculation
        if satellite.satellite_type == SatelliteType.LEO:
            # LEO satellites move quickly
            time_elapsed = (current_time - satellite.last_contact).total_seconds() / 60.0  # minutes
            orbital_progress = (time_elapsed / satellite.period) * 2 * math.pi
            
            # Update position
            satellite.position["lng"] = (satellite.position.get("lng", 0) + orbital_progress * 180 / math.pi) % 360
            satellite.position["lat"] = satellite.position.get("lat", 0) + math.sin(orbital_progress) * satellite.inclination
        
        elif satellite.satellite_type == SatelliteType.GEO:
            # GEO satellites are stationary relative to Earth
            pass  # Position remains constant
    
    async def _space_weather_loop(self) -> None:
        """Space weather monitoring loop."""
        while self.is_running:
            try:
                # Simulate space weather data
                space_weather = SpaceWeather(
                    solar_flux=np.random.normal(100, 20),
                    geomagnetic_index=np.random.exponential(2),
                    proton_flux=np.random.exponential(1),
                    electron_flux=np.random.exponential(5),
                    radiation_level=np.random.normal(0.1, 0.05),
                    aurora_activity=np.random.uniform(0, 10),
                    ionospheric_disturbance=np.random.uniform(0, 1),
                    impact_on_communications=self._assess_communication_impact(),
                    impact_on_satellites=self._assess_satellite_impact()
                )
                
                self.space_weather.append(space_weather)
                
                # Check for space weather alerts
                if space_weather.geomagnetic_index > 5 or space_weather.proton_flux > 10:
                    self.stats['space_weather_alerts'] += 1
                    logger.warning(
                        "Space weather alert",
                        geomagnetic_index=space_weather.geomagnetic_index,
                        proton_flux=space_weather.proton_flux
                    )
                
                await asyncio.sleep(300)  # Update every 5 minutes
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Space weather monitoring error", error=str(e))
                await asyncio.sleep(300)
    
    def _assess_communication_impact(self) -> str:
        """Assess impact of space weather on communications."""
        # Simplified assessment
        if np.random.random() < 0.1:
            return "severe"
        elif np.random.random() < 0.3:
            return "moderate"
        else:
            return "minimal"
    
    def _assess_satellite_impact(self) -> str:
        """Assess impact of space weather on satellites."""
        # Simplified assessment
        if np.random.random() < 0.05:
            return "severe"
        elif np.random.random() < 0.2:
            return "moderate"
        else:
            return "minimal"
    
    async def _communication_monitoring_loop(self) -> None:
        """Communication link monitoring loop."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                for link in self.communication_links.values():
                    # Update link duration
                    link.duration = (current_time - link.established_at).total_seconds() / 60.0
                    
                    # Simulate data transfer
                    if link.link_quality > 0.5:
                        data_transfer = np.random.exponential(link.data_rate * 1000000 / 8)  # bytes
                        link.data_transferred += int(data_transfer)
                        self.stats['total_data_transferred'] += int(data_transfer)
                    
                    # Update link quality based on space weather
                    if self.space_weather:
                        latest_weather = self.space_weather[-1]
                        if latest_weather.impact_on_communications == "severe":
                            link.link_quality *= 0.8
                        elif latest_weather.impact_on_communications == "moderate":
                            link.link_quality *= 0.9
                    
                    # Terminate link if quality is too low
                    if link.link_quality < 0.1:
                        del self.communication_links[link.link_id]
                        self.stats['active_communication_links'] -= 1
                        logger.warning("Communication link terminated due to poor quality", link_id=link.link_id)
                
                # Update average link quality
                if self.communication_links:
                    self.stats['average_link_quality'] = np.mean([link.link_quality for link in self.communication_links.values()])
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Communication monitoring error", error=str(e))
                await asyncio.sleep(30)
    
    def get_satellite_position(self, satellite_id: str) -> Optional[Dict[str, float]]:
        """Get current satellite position."""
        satellite = self.satellites.get(satellite_id)
        return satellite.position if satellite else None
    
    def get_visible_satellites(self, location: Dict[str, float], elevation_threshold: float = 10.0) -> List[str]:
        """Get satellites visible from a location."""
        visible_satellites = []
        
        for satellite_id, satellite in self.satellites.items():
            if satellite.status != "active":
                continue
            
            # Calculate elevation angle (simplified)
            elevation = self._calculate_elevation_angle(location, satellite.position)
            
            if elevation >= elevation_threshold:
                visible_satellites.append(satellite_id)
        
        return visible_satellites
    
    def _calculate_elevation_angle(self, ground_location: Dict[str, float], 
                                 satellite_position: Dict[str, float]) -> float:
        """Calculate elevation angle of satellite from ground location."""
        # Simplified elevation calculation
        lat_diff = satellite_position.get('lat', 0) - ground_location.get('lat', 0)
        lng_diff = satellite_position.get('lng', 0) - ground_location.get('lng', 0)
        alt_diff = satellite_position.get('alt', 0) - ground_location.get('alt', 0)
        
        horizontal_distance = math.sqrt(lat_diff**2 + lng_diff**2)
        elevation = math.degrees(math.atan2(alt_diff, horizontal_distance))
        
        return elevation
    
    def get_communication_links(self, satellite_id: Optional[str] = None) -> List[CommunicationLink]:
        """Get communication links."""
        if satellite_id:
            return [link for link in self.communication_links.values() if link.satellite_id == satellite_id]
        return list(self.communication_links.values())
    
    def get_space_weather(self, limit: int = 10) -> List[SpaceWeather]:
        """Get recent space weather data."""
        return list(self.space_weather)[-limit:] if self.space_weather else []
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            **self.stats,
            'satellites': {
                satellite_id: {
                    'name': satellite.name,
                    'type': satellite.satellite_type.value,
                    'status': satellite.status,
                    'altitude': satellite.altitude,
                    'position': satellite.position,
                    'last_contact': satellite.last_contact.isoformat()
                }
                for satellite_id, satellite in self.satellites.items()
            },
            'ground_stations': {
                station_id: {
                    'name': station.name,
                    'location': station.location,
                    'status': station.status,
                    'data_rate': station.data_rate,
                    'availability': station.availability
                }
                for station_id, station in self.ground_stations.items()
            },
            'communication_links': {
                link_id: {
                    'satellite_id': link.satellite_id,
                    'ground_station_id': link.ground_station_id,
                    'protocol': link.protocol.value,
                    'signal_strength': link.signal_strength,
                    'link_quality': link.link_quality,
                    'data_transferred': link.data_transferred
                }
                for link_id, link in self.communication_links.items()
            },
            'recent_space_weather': [
                weather.to_dict() for weather in self.get_space_weather(5)
            ]
        }

# =============================================================================
# GLOBAL SPACE TECHNOLOGY INSTANCES
# =============================================================================

# Global space technology manager
space_technology_manager = SpaceTechnologyManager()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SatelliteType',
    'OrbitType',
    'CommunicationProtocol',
    'Satellite',
    'GroundStation',
    'CommunicationLink',
    'SpaceWeather',
    'SpaceTechnologyManager',
    'space_technology_manager'
]





























