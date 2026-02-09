"""
Sistema de diario de piel
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class JournalEntry:
    """Entrada del diario"""
    id: str
    user_id: str
    date: str
    mood: Optional[str] = None
    notes: Optional[str] = None
    skin_condition: Optional[str] = None
    products_used: List[str] = None
    photos: List[str] = None
    tags: List[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.products_used is None:
            self.products_used = []
        if self.photos is None:
            self.photos = []
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "date": self.date,
            "mood": self.mood,
            "notes": self.notes,
            "skin_condition": self.skin_condition,
            "products_used": self.products_used,
            "photos": self.photos,
            "tags": self.tags,
            "created_at": self.created_at
        }


class SkinJournal:
    """Sistema de diario de piel"""
    
    def __init__(self):
        """Inicializa el diario"""
        self.entries: Dict[str, List[JournalEntry]] = {}  # user_id -> [entries]
    
    def create_entry(self, user_id: str, date: str, mood: Optional[str] = None,
                    notes: Optional[str] = None, skin_condition: Optional[str] = None,
                    products_used: Optional[List[str]] = None,
                    photos: Optional[List[str]] = None,
                    tags: Optional[List[str]] = None) -> JournalEntry:
        """Crea una nueva entrada"""
        entry = JournalEntry(
            id=str(uuid.uuid4()),
            user_id=user_id,
            date=date,
            mood=mood,
            notes=notes,
            skin_condition=skin_condition,
            products_used=products_used or [],
            photos=photos or [],
            tags=tags or []
        )
        
        if user_id not in self.entries:
            self.entries[user_id] = []
        
        self.entries[user_id].append(entry)
        return entry
    
    def get_user_entries(self, user_id: str, limit: int = 50,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> List[JournalEntry]:
        """Obtiene entradas del usuario"""
        user_entries = self.entries.get(user_id, [])
        
        # Filtrar por fecha
        if start_date:
            user_entries = [e for e in user_entries if e.date >= start_date]
        if end_date:
            user_entries = [e for e in user_entries if e.date <= end_date]
        
        # Ordenar por fecha
        user_entries.sort(key=lambda x: x.date, reverse=True)
        
        # Limitar resultados
        return user_entries[:limit]
    
    def get_entry(self, user_id: str, entry_id: str) -> Optional[JournalEntry]:
        """Obtiene una entrada específica"""
        user_entries = self.entries.get(user_id, [])
        
        for entry in user_entries:
            if entry.id == entry_id:
                return entry
        
        return None
    
    def search_entries(self, user_id: str, query: str) -> List[JournalEntry]:
        """Busca entradas por texto"""
        user_entries = self.entries.get(user_id, [])
        
        query_lower = query.lower()
        matching = []
        
        for entry in user_entries:
            # Buscar en notas
            if entry.notes and query_lower in entry.notes.lower():
                matching.append(entry)
                continue
            
            # Buscar en tags
            if entry.tags:
                for tag in entry.tags:
                    if query_lower in tag.lower():
                        matching.append(entry)
                        break
        
        return matching
    
    def get_journal_statistics(self, user_id: str) -> Dict:
        """Obtiene estadísticas del diario"""
        user_entries = self.entries.get(user_id, [])
        
        total_entries = len(user_entries)
        
        # Productos más usados
        product_counts = {}
        for entry in user_entries:
            for product in entry.products_used:
                product_counts[product] = product_counts.get(product, 0) + 1
        
        top_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Tags más comunes
        tag_counts = {}
        for entry in user_entries:
            for tag in entry.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_entries": total_entries,
            "top_products": [{"product": p, "count": c} for p, c in top_products],
            "top_tags": [{"tag": t, "count": c} for t, c in top_tags],
            "entries_with_photos": len([e for e in user_entries if e.photos])
        }






