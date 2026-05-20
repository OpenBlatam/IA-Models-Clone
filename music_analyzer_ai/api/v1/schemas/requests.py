"""
Request Schemas

Pydantic models for request validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any


class AnalyzeTrackRequest(BaseModel):
    """Request schema for track analysis"""
    track_id: Optional[str] = Field(None, description="Spotify track ID")
    track_name: Optional[str] = Field(None, description="Track name to search")
    include_coaching: bool = Field(False, description="Include coaching analysis")
    
    @validator('track_id', 'track_name')
    def validate_track_identifier(cls, v, values):
        """At least one of track_id or track_name must be provided"""
        if not v and not values.get('track_id') and not values.get('track_name'):
            raise ValueError("Either track_id or track_name must be provided")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "track_id": "4uLU6hMCjMI75M1A2tKUQC",
                "include_coaching": True
            }
        }


class SearchTracksRequest(BaseModel):
    """Request schema for track search"""
    query: str = Field(..., min_length=1, max_length=200, description="Search query")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Pagination offset")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Bohemian Rhapsody",
                "limit": 10,
                "offset": 0
            }
        }


class GeneratePlaylistRequest(BaseModel):
    """Request schema for playlist generation"""
    genres: Optional[List[str]] = Field(None, description="Target genres")
    moods: Optional[List[str]] = Field(None, description="Target moods")
    energy_range: Optional[List[float]] = Field(None, description="Energy range [min, max]")
    tempo_range: Optional[List[int]] = Field(None, description="Tempo range [min, max]")
    seed_track_id: Optional[str] = Field(None, description="Seed track ID")
    length: int = Field(20, ge=1, le=100, description="Playlist length")
    
    @validator('energy_range')
    def validate_energy_range(cls, v):
        """Validate energy range"""
        if v and len(v) != 2:
            raise ValueError("energy_range must have exactly 2 values [min, max]")
        if v and (v[0] < 0 or v[0] > 1 or v[1] < 0 or v[1] > 1 or v[0] > v[1]):
            raise ValueError("energy_range values must be between 0 and 1, and min <= max")
        return v
    
    @validator('tempo_range')
    def validate_tempo_range(cls, v):
        """Validate tempo range"""
        if v and len(v) != 2:
            raise ValueError("tempo_range must have exactly 2 values [min, max]")
        if v and (v[0] < 0 or v[1] < 0 or v[0] > v[1]):
            raise ValueError("tempo_range min must be <= max and both >= 0")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "genres": ["Rock", "Pop"],
                "moods": ["energetic", "happy"],
                "energy_range": [0.6, 1.0],
                "tempo_range": [100, 160],
                "length": 20
            }
        }




