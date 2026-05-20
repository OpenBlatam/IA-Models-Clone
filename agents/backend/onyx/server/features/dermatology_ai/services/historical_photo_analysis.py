"""
Sistema de análisis de fotos históricas
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid


@dataclass
class HistoricalPhoto:
    """Foto histórica"""
    id: str
    user_id: str
    image_url: str
    date: str
    analysis_data: Optional[Dict] = None
    tags: List[str] = None
    notes: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "image_url": self.image_url,
            "date": self.date,
            "analysis_data": self.analysis_data,
            "tags": self.tags,
            "notes": self.notes,
            "created_at": self.created_at
        }


@dataclass
class PhotoTimeline:
    """Timeline de fotos"""
    user_id: str
    photos: List[HistoricalPhoto]
    time_span_days: int
    total_photos: int
    analysis_coverage: float  # Porcentaje de fotos con análisis
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "photos": [p.to_dict() for p in self.photos],
            "time_span_days": self.time_span_days,
            "total_photos": self.total_photos,
            "analysis_coverage": self.analysis_coverage
        }


class HistoricalPhotoAnalysis:
    """Sistema de análisis de fotos históricas"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.photos: Dict[str, List[HistoricalPhoto]] = {}  # user_id -> [photos]
    
    def add_photo(self, user_id: str, image_url: str, date: str,
                 analysis_data: Optional[Dict] = None,
                 tags: Optional[List[str]] = None,
                 notes: Optional[str] = None) -> HistoricalPhoto:
        """Agrega foto histórica"""
        photo = HistoricalPhoto(
            id=str(uuid.uuid4()),
            user_id=user_id,
            image_url=image_url,
            date=date,
            analysis_data=analysis_data,
            tags=tags or [],
            notes=notes
        )
        
        if user_id not in self.photos:
            self.photos[user_id] = []
        
        self.photos[user_id].append(photo)
        return photo
    
    def get_photo_timeline(self, user_id: str, start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> PhotoTimeline:
        """Obtiene timeline de fotos"""
        user_photos = self.photos.get(user_id, [])
        
        # Filtrar por fecha
        if start_date:
            user_photos = [p for p in user_photos if p.date >= start_date]
        if end_date:
            user_photos = [p for p in user_photos if p.date <= end_date]
        
        # Ordenar por fecha
        user_photos.sort(key=lambda x: x.date)
        
        # Calcular span de tiempo
        if len(user_photos) >= 2:
            first_date = datetime.fromisoformat(user_photos[0].date)
            last_date = datetime.fromisoformat(user_photos[-1].date)
            time_span = (last_date - first_date).days
        else:
            time_span = 0
        
        # Calcular cobertura de análisis
        with_analysis = sum(1 for p in user_photos if p.analysis_data)
        analysis_coverage = (with_analysis / len(user_photos) * 100) if user_photos else 0.0
        
        return PhotoTimeline(
            user_id=user_id,
            photos=user_photos,
            time_span_days=time_span,
            total_photos=len(user_photos),
            analysis_coverage=analysis_coverage
        )
    
    def search_photos(self, user_id: str, query: str) -> List[HistoricalPhoto]:
        """Busca fotos por query"""
        user_photos = self.photos.get(user_id, [])
        
        query_lower = query.lower()
        matching = []
        
        for photo in user_photos:
            # Buscar en tags
            if any(query_lower in tag.lower() for tag in photo.tags):
                matching.append(photo)
                continue
            
            # Buscar en notas
            if photo.notes and query_lower in photo.notes.lower():
                matching.append(photo)
        
        return matching
    
    def get_photos_by_date_range(self, user_id: str, days: int = 30) -> List[HistoricalPhoto]:
        """Obtiene fotos en rango de fechas"""
        user_photos = self.photos.get(user_id, [])
        
        cutoff = datetime.now() - timedelta(days=days)
        recent = [p for p in user_photos if datetime.fromisoformat(p.date) >= cutoff]
        
        recent.sort(key=lambda x: x.date, reverse=True)
        return recent






