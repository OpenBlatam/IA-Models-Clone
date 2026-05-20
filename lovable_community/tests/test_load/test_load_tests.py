"""
Tests de carga y stress para Lovable Community
"""

import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from tests.helpers.advanced_helpers import DataFactory, PerformanceHelper
from tests.helpers.test_helpers import generate_user_id


class TestConcurrentRequests:
    """Tests de requests concurrentes"""
    
    @pytest.mark.load
    @pytest.mark.slow
    def test_concurrent_publish_requests(self, chat_service):
        """Test de múltiples publicaciones concurrentes"""
        num_requests = 50
        user_ids = [generate_user_id() for _ in range(10)]
        
        def publish_chat(user_id: str, index: int):
            try:
                return chat_service.publish_chat(
                    user_id=user_id,
                    title=f"Concurrent Chat {index}",
                    chat_content="{}"
                )
            except Exception as e:
                return {"error": str(e)}
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(publish_chat, user_ids[i % len(user_ids)], i)
                for i in range(num_requests)
            ]
            
            results = [future.result() for future in as_completed(futures)]
        
        elapsed = time.time() - start_time
        
        # Verificar que todas se completaron
        successful = [r for r in results if not isinstance(r, dict) or "error" not in r]
        
        assert len(successful) >= num_requests * 0.9, \
            f"Only {len(successful)}/{num_requests} requests succeeded"
        
        assert elapsed < 30.0, \
            f"Took {elapsed:.2f}s, expected < 30s"
    
    @pytest.mark.load
    @pytest.mark.slow
    def test_concurrent_votes(self, chat_service, sample_user_id):
        """Test de votos concurrentes en el mismo chat"""
        # Crear chat
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Vote Stress Test",
            chat_content="{}"
        )
        
        num_votes = 100
        
        def vote(chat_id: str, voter_id: str):
            try:
                return chat_service.vote_chat(chat_id, voter_id, "upvote")
            except Exception as e:
                return {"error": str(e)}
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(vote, chat.id, f"voter-{i}")
                for i in range(num_votes)
            ]
            
            results = [future.result() for future in as_completed(futures)]
        
        elapsed = time.time() - start_time
        
        # Verificar votos
        chat_service.db.refresh(chat)
        
        assert chat.vote_count >= num_votes * 0.9, \
            f"Only {chat.vote_count}/{num_votes} votes recorded"
        
        assert elapsed < 10.0, \
            f"Took {elapsed:.2f}s, expected < 10s"


class TestHighVolumeOperations:
    """Tests de operaciones de alto volumen"""
    
    @pytest.mark.load
    @pytest.mark.slow
    def test_large_search_result(self, chat_service, sample_user_id):
        """Test de búsqueda con muchos resultados"""
        # Crear muchos chats
        num_chats = 500
        
        start = time.time()
        for i in range(num_chats):
            chat_service.publish_chat(
                user_id=sample_user_id,
                title=f"Search Test {i}",
                chat_content="{}",
                tags=["search-test"]
            )
        create_time = time.time() - start
        
        # Buscar
        start = time.time()
        results, total = chat_service.search_chats(
            tags=["search-test"],
            page_size=100
        )
        search_time = time.time() - start
        
        assert total >= num_chats * 0.9, \
            f"Only {total}/{num_chats} chats found"
        
        assert len(results) == 100, \
            f"Expected 100 results, got {len(results)}"
        
        assert search_time < 5.0, \
            f"Search took {search_time:.2f}s, expected < 5s"
        
        assert create_time < 60.0, \
            f"Creation took {create_time:.2f}s, expected < 60s"
    
    @pytest.mark.load
    @pytest.mark.slow
    def test_bulk_operations_large_batch(self, chat_service, sample_user_id):
        """Test de operaciones en lote con muchos elementos"""
        # Crear muchos chats
        chat_ids = []
        for i in range(100):
            chat = chat_service.publish_chat(
                user_id=sample_user_id,
                title=f"Bulk Test {i}",
                chat_content="{}"
            )
            chat_ids.append(chat.id)
        
        # Operación en lote
        start = time.time()
        result = chat_service.bulk_operation(chat_ids, "feature")
        elapsed = time.time() - start
        
        assert result["successful"] >= 90, \
            f"Only {result['successful']}/100 operations succeeded"
        
        assert elapsed < 10.0, \
            f"Bulk operation took {elapsed:.2f}s, expected < 10s"


class TestSustainedLoad:
    """Tests de carga sostenida"""
    
    @pytest.mark.load
    @pytest.mark.slow
    def test_sustained_publish_load(self, chat_service):
        """Test de carga sostenida de publicaciones"""
        duration_seconds = 30
        user_ids = [generate_user_id() for _ in range(5)]
        
        start_time = time.time()
        count = 0
        
        while time.time() - start_time < duration_seconds:
            user_id = user_ids[count % len(user_ids)]
            try:
                chat_service.publish_chat(
                    user_id=user_id,
                    title=f"Sustained Load {count}",
                    chat_content="{}"
                )
                count += 1
            except Exception as e:
                # Continuar en caso de error
                pass
        
        elapsed = time.time() - start_time
        rate = count / elapsed if elapsed > 0 else 0
        
        assert count > 0, "No operations completed"
        assert rate > 1.0, \
            f"Rate too low: {rate:.2f} ops/sec, expected > 1.0"
    
    @pytest.mark.load
    @pytest.mark.slow
    def test_sustained_search_load(self, chat_service, sample_user_id):
        """Test de carga sostenida de búsquedas"""
        # Crear datos base
        for i in range(100):
            chat_service.publish_chat(
                user_id=sample_user_id,
                title=f"Search Load {i}",
                chat_content="{}",
                tags=["load-test"]
            )
        
        duration_seconds = 20
        start_time = time.time()
        count = 0
        
        while time.time() - start_time < duration_seconds:
            try:
                chat_service.search_chats(
                    tags=["load-test"],
                    page_size=20
                )
                count += 1
            except Exception as e:
                pass
        
        elapsed = time.time() - start_time
        rate = count / elapsed if elapsed > 0 else 0
        
        assert count > 0, "No searches completed"
        assert rate > 5.0, \
            f"Search rate too low: {rate:.2f} ops/sec, expected > 5.0"


class TestMemoryUsage:
    """Tests de uso de memoria"""
    
    @pytest.mark.load
    @pytest.mark.slow
    def test_memory_under_stress(self, chat_service, sample_user_id):
        """Test de uso de memoria bajo stress"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Crear muchos chats
        chat_ids = []
        for i in range(1000):
            chat = chat_service.publish_chat(
                user_id=sample_user_id,
                title=f"Memory Test {i}",
                chat_content="{}"
            )
            chat_ids.append(chat.id)
            
            # Verificar memoria cada 100 chats
            if i % 100 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                # No debe aumentar más de 500MB
                assert memory_increase < 500, \
                    f"Memory increased by {memory_increase:.2f}MB, expected < 500MB"
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory
        
        assert total_increase < 1000, \
            f"Total memory increase {total_increase:.2f}MB too high"


class TestPerformanceBenchmarks:
    """Tests de benchmarks de performance"""
    
    @pytest.mark.load
    @pytest.mark.performance
    def test_publish_performance(self, chat_service, sample_user_id):
        """Benchmark de performance de publicación"""
        times = []
        
        for i in range(10):
            start = time.time()
            chat_service.publish_chat(
                user_id=sample_user_id,
                title=f"Perf Test {i}",
                chat_content="{}"
            )
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        assert avg_time < 0.1, \
            f"Average publish time {avg_time:.3f}s too slow, expected < 0.1s"
        
        assert max_time < 0.5, \
            f"Max publish time {max_time:.3f}s too slow, expected < 0.5s"
    
    @pytest.mark.load
    @pytest.mark.performance
    def test_search_performance(self, chat_service, sample_user_id):
        """Benchmark de performance de búsqueda"""
        # Crear datos
        for i in range(100):
            chat_service.publish_chat(
                user_id=sample_user_id,
                title=f"Search Perf {i}",
                chat_content="{}",
                tags=["perf-test"]
            )
        
        times = []
        for i in range(20):
            start = time.time()
            chat_service.search_chats(tags=["perf-test"], page_size=20)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        p95_time = sorted(times)[int(len(times) * 0.95)]
        
        assert avg_time < 0.05, \
            f"Average search time {avg_time:.3f}s too slow, expected < 0.05s"
        
        assert p95_time < 0.1, \
            f"P95 search time {p95_time:.3f}s too slow, expected < 0.1s"

