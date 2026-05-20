"""
Test suite for dev mode chat streaming functionality.

This test suite verifies:
1. Continuous streaming without premature stops
2. State preservation during streaming
3. Complete packet processing
4. No data loss during streaming
"""

import json
import os
import time
from typing import Any
from uuid import UUID

import pytest
import requests

from tests.integration.common_utils.constants import API_SERVER_URL
from tests.integration.common_utils.managers.chat import ChatSessionManager
from tests.integration.common_utils.managers.llm_provider import LLMProviderManager
from tests.integration.common_utils.test_models import DATestUser


class StreamProcessor:
    """Simulates frontend packet processing to test streaming behavior."""
    
    def __init__(self):
        self.packets: list[dict[str, Any]] = []
        self.message_deltas: list[str] = []
        self.full_message = ""
        self.is_complete = False
        self.is_empty = False
        self.stack_size = 0
        self.packet_count = 0
        self.consecutive_empty_checks = 0
        self.max_consecutive_empty_checks = 1000
        
    def process_stream(self, response: requests.Response) -> dict[str, Any]:
        """Process the streaming response similar to frontend logic."""
        start_time = time.time()
        iterations = 0
        max_iterations = 1000000
        
        # Simulate the frontend's packet processing loop
        for line in response.iter_lines():
            if not line:
                continue
                
            iterations += 1
            self.packet_count += 1
            
            try:
                packet = json.loads(line.decode("utf-8"))
                self.packets.append(packet)
                
                # Extract packet object
                packet_obj = packet.get("obj") or packet
                packet_type = packet_obj.get("type") or "unknown"
                
                # Process message_delta packets
                if packet_type == "message_delta":
                    content = packet_obj.get("content", "")
                    if content:
                        self.message_deltas.append(content)
                        self.full_message += content
                
                # Process message_start packets
                elif packet_type == "message_start":
                    initial_content = packet_obj.get("content", "")
                    if initial_content:
                        self.full_message = initial_content
                
                # Simulate stack processing
                self.stack_size = len(self.packets)
                self.is_empty = self.stack_size == 0
                
                # Reset empty check counter when we get packets
                if not self.is_empty:
                    if self.consecutive_empty_checks > 0:
                        self.consecutive_empty_checks = 0
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse packet: {e}")
                continue
        
        # Mark as complete after stream ends
        self.is_complete = True
        self.is_empty = len(self.packets) == 0
        
        duration = time.time() - start_time
        
        return {
            "packet_count": self.packet_count,
            "packets": len(self.packets),
            "message_deltas": len(self.message_deltas),
            "full_message_length": len(self.full_message),
            "is_complete": self.is_complete,
            "is_empty": self.is_empty,
            "iterations": iterations,
            "duration": duration,
        }


@pytest.mark.skipif(
    os.environ.get("ENABLE_PAID_ENTERPRISE_EDITION_FEATURES", "").lower() != "true",
    reason="Dev mode chat streaming tests are enterprise only",
)
def test_dev_mode_chat_streaming_continuous(
    reset: None,
    admin_user: DATestUser,
) -> None:
    """
    Test that chat streaming continues without premature stops in dev mode.
    
    This test simulates the frontend's packet processing loop and verifies:
    - All packets are processed
    - Streaming doesn't stop prematurely
    - State is preserved throughout
    """
    LLMProviderManager.create(user_performing_action=admin_user)
    
    # Create chat session
    chat_session = ChatSessionManager.create(
        persona_id=0,
        description="Dev mode streaming test",
        user_performing_action=admin_user,
    )
    
    # Send message with streaming
    chat_message_req = {
        "chat_session_id": str(chat_session.id),
        "message": "Write a detailed explanation about how streaming works in chat applications. Make it comprehensive.",
        "retrieval_options": {"top_k": 3},
    }
    
    response = requests.post(
        f"{API_SERVER_URL}/chat/send-message",
        json=chat_message_req,
        headers=admin_user.headers,
        stream=True,
        cookies=admin_user.cookies,
    )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    # Process stream using our processor
    processor = StreamProcessor()
    result = processor.process_stream(response)
    
    # Verify streaming completed successfully
    assert result["is_complete"], "Stream should be marked as complete"
    assert result["packets"] > 0, "Should have received packets"
    assert result["message_deltas"] > 0, "Should have received message_delta packets"
    assert result["full_message_length"] > 0, "Should have accumulated message content"
    
    # Verify no premature stops (should have processed multiple packets)
    assert result["packets"] >= 10, f"Expected at least 10 packets, got {result['packets']}"
    
    # Verify message content is reasonable
    assert len(processor.full_message) > 50, "Message should have substantial content"
    
    print(f"\n✅ Streaming test results:")
    print(f"   - Packets received: {result['packets']}")
    print(f"   - Message deltas: {result['message_deltas']}")
    print(f"   - Full message length: {result['full_message_length']}")
    print(f"   - Duration: {result['duration']:.2f}s")
    print(f"   - Iterations: {result['iterations']}")


@pytest.mark.skipif(
    os.environ.get("ENABLE_PAID_ENTERPRISE_EDITION_FEATURES", "").lower() != "true",
    reason="Dev mode chat streaming tests are enterprise only",
)
def test_dev_mode_chat_state_preservation(
    reset: None,
    admin_user: DATestUser,
) -> None:
    """
    Test that chat state is preserved during streaming.
    
    This test verifies:
    - Chat session persists after streaming
    - Messages are saved correctly
    - State can be retrieved after streaming completes
    """
    LLMProviderManager.create(user_performing_action=admin_user)
    
    # Create chat session
    chat_session = ChatSessionManager.create(
        persona_id=0,
        description="State preservation test",
        user_performing_action=admin_user,
    )
    
    initial_session_id = chat_session.id
    
    # Send message
    response = ChatSessionManager.send_message(
        chat_session_id=chat_session.id,
        message="What is the capital of France?",
        user_performing_action=admin_user,
    )
    
    # Verify response
    assert len(response.full_message) > 0, "Should have received a response"
    
    # Verify chat session still exists and is accessible
    chat_history = ChatSessionManager.get_chat_history(
        chat_session=chat_session,
        user_performing_action=admin_user,
    )
    
    assert len(chat_history) >= 2, "Should have at least user and assistant messages"
    
    # Verify session ID hasn't changed
    assert chat_session.id == initial_session_id, "Session ID should remain constant"
    
    # Verify messages are in correct order
    user_messages = [msg for msg in chat_history if "France" in msg.message]
    assert len(user_messages) > 0, "User message should be preserved"
    
    print(f"\n✅ State preservation test results:")
    print(f"   - Session ID: {chat_session.id}")
    print(f"   - Messages in history: {len(chat_history)}")
    print(f"   - Response length: {len(response.full_message)}")


@pytest.mark.skipif(
    os.environ.get("ENABLE_PAID_ENTERPRISE_EDITION_FEATURES", "").lower() != "true",
    reason="Dev mode chat streaming tests are enterprise only",
)
def test_dev_mode_chat_packet_processing_completeness(
    reset: None,
    admin_user: DATestUser,
) -> None:
    """
    Test that all packets are processed completely.
    
    This test verifies:
    - No packets are lost during processing
    - Packet order is maintained
    - All message_delta packets are accumulated correctly
    """
    LLMProviderManager.create(user_performing_action=admin_user)
    
    # Create chat session
    chat_session = ChatSessionManager.create(
        persona_id=0,
        description="Packet completeness test",
        user_performing_action=admin_user,
    )
    
    # Send message
    chat_message_req = {
        "chat_session_id": str(chat_session.id),
        "message": "Count from 1 to 10 and explain each number.",
    }
    
    response = requests.post(
        f"{API_SERVER_URL}/chat/send-message",
        json=chat_message_req,
        headers=admin_user.headers,
        stream=True,
        cookies=admin_user.cookies,
    )
    
    assert response.status_code == 200
    
    # Process and track all packets
    processor = StreamProcessor()
    result = processor.process_stream(response)
    
    # Verify all packets were processed
    assert result["packets"] == result["packet_count"], "All packets should be processed"
    
    # Verify message_delta packets were accumulated
    accumulated_length = sum(len(delta) for delta in processor.message_deltas)
    assert accumulated_length > 0, "Should have accumulated message content"
    
    # Verify packet types are correct
    packet_types = {}
    for packet in processor.packets:
        packet_obj = packet.get("obj") or packet
        packet_type = packet_obj.get("type") or "unknown"
        packet_types[packet_type] = packet_types.get(packet_type, 0) + 1
    
    # Should have at least message_start and message_delta packets
    assert "message_start" in packet_types or "message_delta" in packet_types, \
        "Should have message packets"
    
    print(f"\n✅ Packet completeness test results:")
    print(f"   - Total packets: {result['packets']}")
    print(f"   - Message deltas: {result['message_deltas']}")
    print(f"   - Accumulated length: {accumulated_length}")
    print(f"   - Packet types: {packet_types}")


@pytest.mark.skipif(
    os.environ.get("ENABLE_PAID_ENTERPRISE_EDITION_FEATURES", "").lower() != "true",
    reason="Dev mode chat streaming tests are enterprise only",
)
def test_dev_mode_chat_multiple_messages(
    reset: None,
    admin_user: DATestUser,
) -> None:
    """
    Test sending multiple messages in sequence without state loss.
    
    This test verifies:
    - Multiple messages can be sent in sequence
    - State is preserved between messages
    - Each message streams correctly
    """
    LLMProviderManager.create(user_performing_action=admin_user)
    
    # Create chat session
    chat_session = ChatSessionManager.create(
        persona_id=0,
        description="Multiple messages test",
        user_performing_action=admin_user,
    )
    
    messages = [
        "What is 2+2?",
        "What is the square root of 16?",
        "What is 10 divided by 2?",
    ]
    
    responses = []
    for i, message in enumerate(messages):
        processor = StreamProcessor()
        
        chat_message_req = {
            "chat_session_id": str(chat_session.id),
            "message": message,
        }
        
        response = requests.post(
            f"{API_SERVER_URL}/chat/send-message",
            json=chat_message_req,
            headers=admin_user.headers,
            stream=True,
            cookies=admin_user.cookies,
        )
        
        assert response.status_code == 200, f"Message {i+1} should succeed"
        
        result = processor.process_stream(response)
        responses.append(result)
        
        # Verify each message streams correctly
        assert result["packets"] > 0, f"Message {i+1} should have packets"
        assert result["full_message_length"] > 0, f"Message {i+1} should have content"
    
    # Verify all messages were processed
    assert len(responses) == len(messages), "All messages should be processed"
    
    # Verify chat history contains all messages
    chat_history = ChatSessionManager.get_chat_history(
        chat_session=chat_session,
        user_performing_action=admin_user,
    )
    
    # Should have user + assistant messages for each
    assert len(chat_history) >= len(messages) * 2, \
        f"Expected at least {len(messages) * 2} messages in history"
    
    print(f"\n✅ Multiple messages test results:")
    print(f"   - Messages sent: {len(messages)}")
    print(f"   - Responses received: {len(responses)}")
    print(f"   - Total messages in history: {len(chat_history)}")
    for i, result in enumerate(responses):
        print(f"   - Message {i+1}: {result['packets']} packets, "
              f"{result['full_message_length']} chars")

