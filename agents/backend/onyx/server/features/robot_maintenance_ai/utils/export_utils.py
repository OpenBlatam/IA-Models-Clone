"""
Utilities for exporting conversation data and reports.
"""

import csv
from typing import Dict, List, Any
from pathlib import Path
import logging

from .file_helpers import ensure_directory_exists, write_json_file, get_iso_timestamp
from .data_helpers import count_matching

logger = logging.getLogger(__name__)


def export_conversation_json(conversation: List[Dict[str, Any]], output_path: str) -> str:
    """
    Export conversation to JSON file.
    
    Args:
        conversation: List of conversation messages
        output_path: Path to save the JSON file
    
    Returns:
        Path to the exported file
    """
    export_data = {
        "exported_at": get_iso_timestamp(),
        "message_count": len(conversation),
        "messages": conversation
    }
    
    output_file = write_json_file(export_data, output_path)
    logger.info(f"Conversation exported to {output_file}")
    return output_file


def export_conversation_csv(conversation: List[Dict[str, Any]], output_path: str) -> str:
    """
    Export conversation to CSV file.
    
    Args:
        conversation: List of conversation messages
        output_path: Path to save the CSV file
    
    Returns:
        Path to the exported file
    """
    output_file = ensure_directory_exists(output_path)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if not conversation:
            return str(output_file)
        
        fieldnames = ['timestamp', 'role', 'content']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for message in conversation:
            writer.writerow({
                'timestamp': message.get('timestamp', ''),
                'role': message.get('role', ''),
                'content': message.get('content', '')
            })
    
    logger.info(f"Conversation exported to {output_file}")
    return str(output_file)


def generate_maintenance_report(
    conversation: List[Dict[str, Any]],
    robot_type: str,
    maintenance_type: str
) -> Dict[str, Any]:
    """
    Generate a maintenance report from conversation.
    
    Args:
        conversation: List of conversation messages
        robot_type: Type of robot
        maintenance_type: Type of maintenance
    
    Returns:
        Report dictionary
    """
    report = {
        "generated_at": get_iso_timestamp(),
        "robot_type": robot_type,
        "maintenance_type": maintenance_type,
        "total_messages": len(conversation),
        "summary": {
            "questions_asked": count_matching(conversation, lambda msg: msg.get('role') == 'user'),
            "answers_provided": count_matching(conversation, lambda msg: msg.get('role') == 'assistant'),
            "topics_discussed": []
        },
        "conversation": conversation
    }
    
    return report






