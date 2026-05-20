"""
Tests de búsqueda avanzada
"""

import pytest
from unittest.mock import Mock
import re


class TestSearch:
    """Tests de búsqueda"""
    
    def test_basic_search(self):
        """Test de búsqueda básica"""
        def search(query, items):
            query_lower = query.lower()
            results = []
            
            for item in items:
                if query_lower in item.get("name", "").lower():
                    results.append(item)
            
            return results
        
        items = [
            {"id": "1", "name": "Rock Song"},
            {"id": "2", "name": "Pop Song"},
            {"id": "3", "name": "Rock Anthem"}
        ]
        
        results = search("rock", items)
        
        assert len(results) == 2
        assert results[0]["name"] == "Rock Song"
        assert results[1]["name"] == "Rock Anthem"
    
    def test_fuzzy_search(self):
        """Test de búsqueda difusa"""
        def fuzzy_search(query, items, threshold=0.6):
            results = []
            query_lower = query.lower()
            
            for item in items:
                name_lower = item.get("name", "").lower()
                
                # Simulación simple de similitud
                if query_lower in name_lower:
                    similarity = len(query_lower) / max(len(name_lower), 1)
                    if similarity >= threshold:
                        results.append({
                            **item,
                            "similarity": similarity
                        })
            
            return sorted(results, key=lambda x: x["similarity"], reverse=True)
        
        items = [
            {"id": "1", "name": "Rock Song"},
            {"id": "2", "name": "Rock Anthem"},
            {"id": "3", "name": "Pop Song"}
        ]
        
        results = fuzzy_search("rock", items)
        
        assert len(results) > 0
        assert "similarity" in results[0]
    
    def test_search_with_filters(self):
        """Test de búsqueda con filtros"""
        def search_with_filters(query, items, filters=None):
            results = []
            query_lower = query.lower()
            
            for item in items:
                # Filtrar por query
                if query_lower not in item.get("name", "").lower():
                    continue
                
                # Aplicar filtros
                if filters:
                    match = True
                    if "genre" in filters and item.get("genre") != filters["genre"]:
                        match = False
                    if "year" in filters and item.get("year") != filters["year"]:
                        match = False
                    
                    if not match:
                        continue
                
                results.append(item)
            
            return results
        
        items = [
            {"id": "1", "name": "Rock Song", "genre": "Rock", "year": 2020},
            {"id": "2", "name": "Rock Anthem", "genre": "Rock", "year": 2021},
            {"id": "3", "name": "Pop Song", "genre": "Pop", "year": 2020}
        ]
        
        results = search_with_filters("rock", items, {"genre": "Rock", "year": 2020})
        
        assert len(results) == 1
        assert results[0]["id"] == "1"


class TestAdvancedSearch:
    """Tests de búsqueda avanzada"""
    
    def test_boolean_search(self):
        """Test de búsqueda booleana"""
        def boolean_search(query, items):
            # Parsear operadores AND, OR, NOT
            if " AND " in query:
                terms = [t.strip() for t in query.split(" AND ")]
                results = items
                for term in terms:
                    results = [item for item in results 
                              if term.lower() in item.get("name", "").lower()]
                return results
            elif " OR " in query:
                terms = [t.strip() for t in query.split(" OR ")]
                results = []
                for term in terms:
                    results.extend([item for item in items 
                                  if term.lower() in item.get("name", "").lower()])
                return list({item["id"]: item for item in results}.values())
            else:
                return [item for item in items 
                       if query.lower() in item.get("name", "").lower()]
        
        items = [
            {"id": "1", "name": "Rock Song"},
            {"id": "2", "name": "Pop Song"},
            {"id": "3", "name": "Rock Anthem"}
        ]
        
        results_and = boolean_search("Rock AND Song", items)
        assert len(results_and) == 1
        
        results_or = boolean_search("Rock OR Pop", items)
        assert len(results_or) == 3
    
    def test_search_ranking(self):
        """Test de ranking de resultados"""
        def search_with_ranking(query, items):
            query_lower = query.lower()
            results = []
            
            for item in items:
                name_lower = item.get("name", "").lower()
                
                if query_lower in name_lower:
                    # Calcular score
                    score = 0
                    if name_lower.startswith(query_lower):
                        score += 10  # Bonus por coincidencia al inicio
                    score += len(query_lower) / len(name_lower) * 5
                    
                    results.append({
                        **item,
                        "score": score
                    })
            
            return sorted(results, key=lambda x: x["score"], reverse=True)
        
        items = [
            {"id": "1", "name": "Rock Song"},
            {"id": "2", "name": "Song Rock"},
            {"id": "3", "name": "Rock Anthem"}
        ]
        
        results = search_with_ranking("rock", items)
        
        assert len(results) == 3
        # El que empieza con "rock" debería tener mayor score
        assert results[0]["name"] in ["Rock Song", "Rock Anthem"]
    
    def test_search_pagination(self):
        """Test de paginación de resultados"""
        def search_with_pagination(query, items, page=1, per_page=10):
            # Filtrar resultados
            filtered = [item for item in items 
                       if query.lower() in item.get("name", "").lower()]
            
            # Calcular paginación
            total = len(filtered)
            start = (page - 1) * per_page
            end = start + per_page
            
            return {
                "results": filtered[start:end],
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": total,
                    "total_pages": (total + per_page - 1) // per_page
                }
            }
        
        items = [{"id": str(i), "name": f"Item {i}"} for i in range(25)]
        
        result = search_with_pagination("item", items, page=1, per_page=10)
        
        assert len(result["results"]) == 10
        assert result["pagination"]["total"] == 25
        assert result["pagination"]["total_pages"] == 3


class TestSearchIndexing:
    """Tests de indexación de búsqueda"""
    
    def test_build_search_index(self):
        """Test de construcción de índice de búsqueda"""
        def build_index(items):
            index = {}
            
            for item in items:
                name = item.get("name", "").lower()
                words = name.split()
                
                for word in words:
                    if word not in index:
                        index[word] = []
                    if item["id"] not in index[word]:
                        index[word].append(item["id"])
            
            return index
        
        items = [
            {"id": "1", "name": "Rock Song"},
            {"id": "2", "name": "Pop Song"},
            {"id": "3", "name": "Rock Anthem"}
        ]
        
        index = build_index(items)
        
        assert "rock" in index
        assert "song" in index
        assert len(index["rock"]) == 2
        assert len(index["song"]) == 2
    
    def test_search_with_index(self):
        """Test de búsqueda usando índice"""
        def search_with_index(query, index, items_map):
            query_lower = query.lower()
            words = query_lower.split()
            
            result_ids = set()
            for word in words:
                if word in index:
                    result_ids.update(index[word])
            
            return [items_map[id] for id in result_ids]
        
        index = {
            "rock": ["1", "3"],
            "song": ["1", "2"],
            "pop": ["2"]
        }
        
        items_map = {
            "1": {"id": "1", "name": "Rock Song"},
            "2": {"id": "2", "name": "Pop Song"},
            "3": {"id": "3", "name": "Rock Anthem"}
        }
        
        results = search_with_index("rock", index, items_map)
        
        assert len(results) == 2
        assert results[0]["id"] in ["1", "3"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

