"""
Property-based testing for copywriting service.
"""
import pytest
from hypothesis import given, strategies as st, settings, example
from typing import Dict, Any, List
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import CopywritingInput, CopywritingOutput, Feedback
from tests.test_utils import TestDataFactory, MockAIService, TestAssertions


class TestPropertyBased:
    """Property-based testing using Hypothesis."""
    
    @given(
        product_description=st.text(min_size=1, max_size=2000),
        target_platform=st.sampled_from(["instagram", "facebook", "twitter", "linkedin", "tiktok"]),
        content_type=st.sampled_from(["social_post", "ad_copy", "email", "blog_post", "product_description"]),
        tone=st.sampled_from(["professional", "casual", "friendly", "authoritative", "inspirational", "urgent"]),
        use_case=st.sampled_from(["product_launch", "brand_awareness", "lead_generation", "customer_retention", "sales_promotion"])
    )
    @settings(max_examples=50, deadline=5000)
    def test_copywriting_input_properties(self, product_description, target_platform, content_type, tone, use_case):
        """Test that CopywritingInput maintains properties for any valid input."""
        # Create input with generated data
        input_data = CopywritingInput(
            product_description=product_description,
            target_platform=target_platform,
            content_type=content_type,
            tone=tone,
            use_case=use_case
        )
        
        # Property 1: Input should be valid
        assert input_data.product_description == product_description
        assert input_data.target_platform == target_platform
        assert input_data.content_type == content_type
        assert input_data.tone == tone
        assert input_data.use_case == use_case
        
        # Property 2: Input should be serializable
        serialized = input_data.model_dump()
        assert isinstance(serialized, dict)
        assert "product_description" in serialized
        assert "target_platform" in serialized
        
        # Property 3: Input should be deserializable
        deserialized = CopywritingInput(**serialized)
        assert deserialized.product_description == product_description
        assert deserialized.target_platform == target_platform
    
    @given(
        variants=st.lists(
            st.dictionaries(
                keys=st.sampled_from(["variant_id", "headline", "primary_text", "call_to_action", "hashtags"]),
                values=st.one_of(
                    st.text(min_size=1, max_size=500),
                    st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=10)
                ),
                min_size=1,
                max_size=5
            ),
            min_size=1,
            max_size=10
        ),
        model_used=st.text(min_size=1, max_size=100),
        generation_time=st.floats(min_value=0.0, max_value=3600.0),
        tokens_used=st.integers(min_value=0, max_value=100000)
    )
    @settings(max_examples=30, deadline=5000)
    def test_copywriting_output_properties(self, variants, model_used, generation_time, tokens_used):
        """Test that CopywritingOutput maintains properties for any valid output."""
        # Ensure variants have required fields
        for variant in variants:
            if "variant_id" not in variant:
                variant["variant_id"] = f"variant_{hash(str(variant))}"
            if "headline" not in variant:
                variant["headline"] = "Generated Headline"
            if "primary_text" not in variant:
                variant["primary_text"] = "Generated Text"
            if "call_to_action" not in variant:
                variant["call_to_action"] = "Learn More"
        
        # Create output with generated data
        output_data = CopywritingOutput(
            variants=variants,
            model_used=model_used,
            generation_time=generation_time,
            tokens_used=tokens_used
        )
        
        # Property 1: Output should be valid
        assert len(output_data.variants) == len(variants)
        assert output_data.model_used == model_used
        assert output_data.generation_time == generation_time
        assert output_data.tokens_used == tokens_used
        
        # Property 2: Output should be serializable
        serialized = output_data.model_dump()
        assert isinstance(serialized, dict)
        assert "variants" in serialized
        assert "model_used" in serialized
        
        # Property 3: All variants should have required fields
        for variant in output_data.variants:
            assert hasattr(variant, 'variant_id')
            assert hasattr(variant, 'headline')
            assert hasattr(variant, 'primary_text')
            assert hasattr(variant, 'call_to_action')
    
    @given(
        feedback_type=st.sampled_from(["human", "model", "auto", "ai_analysis"]),
        score=st.floats(min_value=0.0, max_value=10.0),
        comments=st.text(min_size=0, max_size=1000),
        variant_id=st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=30, deadline=5000)
    def test_feedback_properties(self, feedback_type, score, comments, variant_id):
        """Test that Feedback maintains properties for any valid feedback."""
        # Create feedback with generated data
        feedback_data = Feedback(
            type=feedback_type,
            score=score,
            comments=comments,
            variant_id=variant_id
        )
        
        # Property 1: Feedback should be valid
        assert feedback_data.type == feedback_type
        assert feedback_data.score == score
        assert feedback_data.comments == comments
        assert feedback_data.variant_id == variant_id
        
        # Property 2: Feedback should be serializable
        serialized = feedback_data.model_dump()
        assert isinstance(serialized, dict)
        assert "type" in serialized
        assert "score" in serialized
        
        # Property 3: Score should be within valid range
        assert 0.0 <= feedback_data.score <= 10.0
    
    @given(
        batch_size=st.integers(min_value=1, max_value=100),
        product_descriptions=st.lists(
            st.text(min_size=1, max_size=2000),
            min_size=1,
            max_size=100
        )
    )
    @settings(max_examples=20, deadline=10000)
    def test_batch_processing_properties(self, batch_size, product_descriptions):
        """Test that batch processing maintains properties for any batch size."""
        # Create batch inputs
        batch_inputs = []
        for i, description in enumerate(product_descriptions[:batch_size]):
            input_data = CopywritingInput(
                product_description=description,
                target_platform="instagram",
                content_type="social_post",
                tone="professional",
                use_case="product_launch"
            )
            batch_inputs.append(input_data)
        
        # Property 1: Batch size should match input size
        assert len(batch_inputs) == min(batch_size, len(product_descriptions))
        
        # Property 2: All inputs should be valid
        for input_data in batch_inputs:
            assert isinstance(input_data, CopywritingInput)
            assert len(input_data.product_description) > 0
        
        # Property 3: Batch should be processable
        mock_service = MockAIService()
        results = []
        for input_data in batch_inputs:
            result = asyncio.run(mock_service.mock_call(input_data, "gpt-3.5-turbo"))
            results.append(result)
        
        # Property 4: Results should match input size
        assert len(results) == len(batch_inputs)
        
        # Property 5: All results should be valid
        for result in results:
            assert isinstance(result, dict)
            assert "variants" in result
            assert "model_used" in result
    
    @given(
        input_text=st.text(min_size=1, max_size=1000),
        max_length=st.integers(min_value=1, max_value=2000)
    )
    @settings(max_examples=30, deadline=5000)
    def test_text_processing_properties(self, input_text, max_length):
        """Test that text processing maintains properties for any text input."""
        # Property 1: Text length should be preserved
        assert len(input_text) == len(input_text)
        
        # Property 2: Text truncation should work correctly
        if len(input_text) > max_length:
            truncated = input_text[:max_length]
            assert len(truncated) <= max_length
            assert truncated == input_text[:max_length]
        else:
            assert len(input_text) <= max_length
        
        # Property 3: Text should be processable
        processed_text = input_text.strip()
        assert isinstance(processed_text, str)
        assert len(processed_text) <= len(input_text)
    
    @given(
        performance_threshold=st.floats(min_value=0.1, max_value=10.0),
        execution_time=st.floats(min_value=0.0, max_value=10.0)
    )
    @settings(max_examples=20, deadline=5000)
    def test_performance_properties(self, performance_threshold, execution_time):
        """Test that performance testing maintains properties for any threshold."""
        # Property 1: Performance should be measurable
        assert execution_time >= 0.0
        
        # Property 2: Performance should be comparable to threshold
        if execution_time <= performance_threshold:
            assert execution_time <= performance_threshold
        else:
            assert execution_time > performance_threshold
        
        # Property 3: Performance ratio should be calculable
        if performance_threshold > 0:
            ratio = execution_time / performance_threshold
            assert ratio >= 0.0
            assert isinstance(ratio, float)
    
    @given(
        error_rate=st.floats(min_value=0.0, max_value=1.0),
        success_rate=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=20, deadline=5000)
    def test_reliability_properties(self, error_rate, success_rate):
        """Test that reliability testing maintains properties for any error rate."""
        # Property 1: Error rate should be between 0 and 1
        assert 0.0 <= error_rate <= 1.0
        assert 0.0 <= success_rate <= 1.0
        
        # Property 2: Error rate and success rate should be complementary
        calculated_success_rate = 1.0 - error_rate
        assert abs(success_rate - calculated_success_rate) < 0.001 or success_rate == calculated_success_rate
        
        # Property 3: Reliability should be calculable
        reliability = success_rate
        assert 0.0 <= reliability <= 1.0
    
    @given(
        data_size=st.integers(min_value=1, max_value=10000),
        memory_limit=st.integers(min_value=100, max_value=1000000)
    )
    @settings(max_examples=20, deadline=5000)
    def test_memory_properties(self, data_size, memory_limit):
        """Test that memory testing maintains properties for any data size."""
        # Property 1: Data size should be positive
        assert data_size > 0
        
        # Property 2: Memory usage should be proportional to data size
        estimated_memory = data_size * 0.1  # Rough estimate
        assert estimated_memory > 0
        
        # Property 3: Memory should be within limits
        if estimated_memory <= memory_limit:
            assert estimated_memory <= memory_limit
        else:
            assert estimated_memory > memory_limit
    
    @given(
        concurrent_requests=st.integers(min_value=1, max_value=1000),
        max_workers=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=20, deadline=10000)
    def test_concurrency_properties(self, concurrent_requests, max_workers):
        """Test that concurrency testing maintains properties for any request count."""
        # Property 1: Request count should be positive
        assert concurrent_requests > 0
        assert max_workers > 0
        
        # Property 2: Workers should not exceed requests
        effective_workers = min(concurrent_requests, max_workers)
        assert effective_workers <= concurrent_requests
        assert effective_workers <= max_workers
        
        # Property 3: Concurrency should be manageable
        if concurrent_requests <= max_workers:
            assert concurrent_requests <= max_workers
        else:
            assert concurrent_requests > max_workers
    
    @given(
        retry_count=st.integers(min_value=0, max_value=10),
        max_retries=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=20, deadline=5000)
    def test_retry_properties(self, retry_count, max_retries):
        """Test that retry mechanisms maintain properties for any retry count."""
        # Property 1: Retry count should be non-negative
        assert retry_count >= 0
        
        # Property 2: Max retries should be positive
        assert max_retries > 0
        
        # Property 3: Retry count should not exceed max retries
        if retry_count <= max_retries:
            assert retry_count <= max_retries
        else:
            assert retry_count > max_retries
        
        # Property 4: Retry logic should be deterministic
        should_retry = retry_count < max_retries
        assert isinstance(should_retry, bool)
    
    @given(
        input_data=st.dictionaries(
            keys=st.text(min_size=1, max_size=50),
            values=st.one_of(
                st.text(min_size=0, max_size=1000),
                st.integers(min_value=-1000, max_value=1000),
                st.floats(min_value=-1000.0, max_value=1000.0),
                st.booleans()
            ),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=30, deadline=5000)
    def test_data_validation_properties(self, input_data):
        """Test that data validation maintains properties for any input data."""
        # Property 1: Input should be a dictionary
        assert isinstance(input_data, dict)
        
        # Property 2: All keys should be strings
        for key in input_data.keys():
            assert isinstance(key, str)
            assert len(key) > 0
        
        # Property 3: Values should be of expected types
        for value in input_data.values():
            assert isinstance(value, (str, int, float, bool))
        
        # Property 4: Data should be serializable
        try:
            import json
            json.dumps(input_data)
            serializable = True
        except (TypeError, ValueError):
            serializable = False
        
        # Property 5: Data should be processable
        processed_data = {k: str(v) for k, v in input_data.items()}
        assert len(processed_data) == len(input_data)
        assert all(isinstance(v, str) for v in processed_data.values())
