"""Example usage of Markdown to Professional Documents AI"""
import requests
import json

# Base URL
BASE_URL = "http://localhost:8035"

def example_convert_to_excel():
    """Example: Convert Markdown to Excel"""
    markdown_content = """
# Sales Report Q1 2024

## Overview
This report shows the sales performance for Q1 2024.

## Sales Data

| Product | Sales | Units | Growth |
|---------|-------|-------|--------|
| Widget A | $10,000 | 100 | 15% |
| Widget B | $15,000 | 150 | 20% |
| Widget C | $8,000 | 80 | 10% |
| Widget D | $12,000 | 120 | 18% |

## Summary
Total sales: $45,000 with an average growth of 15.75%.
    """
    
    response = requests.post(
        f"{BASE_URL}/convert",
        json={
            "markdown_content": markdown_content,
            "output_format": "excel",
            "include_charts": True,
            "include_tables": True
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! Output file: {result['output_file']}")
        print(f"   File size: {result['file_size']} bytes")
        return result
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None


def example_convert_to_pdf():
    """Example: Convert Markdown to PDF"""
    markdown_content = """
# Project Documentation

## Introduction
This document describes the project architecture and implementation.

## Features
- Feature 1: Description
- Feature 2: Description
- Feature 3: Description

## Data Table

| Metric | Value | Status |
|--------|-------|--------|
| Users | 1,000 | Active |
| Revenue | $50,000 | Growing |
| Growth | 25% | Positive |
    """
    
    response = requests.post(
        f"{BASE_URL}/convert",
        json={
            "markdown_content": markdown_content,
            "output_format": "pdf",
            "include_charts": True,
            "include_tables": True
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! Output file: {result['output_file']}")
        return result
    else:
        print(f"❌ Error: {response.status_code}")
        return None


def example_convert_file():
    """Example: Convert a Markdown file"""
    file_path = "example.md"
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {
            'output_format': 'word',
            'include_charts': 'true',
            'include_tables': 'true'
        }
        
        response = requests.post(
            f"{BASE_URL}/convert/file",
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        # Save the response file
        output_filename = "output.docx"
        with open(output_filename, 'wb') as f:
            f.write(response.content)
        print(f"✅ Success! File saved as {output_filename}")
    else:
        print(f"❌ Error: {response.status_code}")


def example_batch_convert():
    """Example: Batch conversion"""
    markdown_contents = [
        "# Document 1\n\n| A | B |\n|---|---|\n| 1 | 2 |",
        "# Document 2\n\n| X | Y |\n|---|---|\n| 3 | 4 |"
    ]
    
    response = requests.post(
        f"{BASE_URL}/convert/batch",
        json={
            "markdown_contents": markdown_contents,
            "output_format": "html",
            "include_charts": True,
            "include_tables": True
        }
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"✅ Success! Converted {len(results)} documents")
        for idx, result in enumerate(results, 1):
            if result['success']:
                print(f"   {idx}. {result['output_file']}")
        return results
    else:
        print(f"❌ Error: {response.status_code}")
        return None


def example_get_formats():
    """Example: Get supported formats"""
    response = requests.get(f"{BASE_URL}/formats")
    
    if response.status_code == 200:
        formats = response.json()
        print("📋 Supported Formats:")
        for format_name, format_info in formats['formats'].items():
            print(f"   • {format_name.upper()}: {format_info['description']}")
        return formats
    else:
        print(f"❌ Error: {response.status_code}")
        return None


if __name__ == "__main__":
    print("🚀 Markdown to Professional Documents AI - Examples\n")
    
    # Check health
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Server is running\n")
        else:
            print("❌ Server is not responding\n")
            exit(1)
    except:
        print("❌ Cannot connect to server. Make sure it's running on port 8035\n")
        exit(1)
    
    # Get formats
    print("=" * 50)
    example_get_formats()
    print()
    
    # Convert to Excel
    print("=" * 50)
    print("Example 1: Convert to Excel")
    example_convert_to_excel()
    print()
    
    # Convert to PDF
    print("=" * 50)
    print("Example 2: Convert to PDF")
    example_convert_to_pdf()
    print()
    
    # Batch convert
    print("=" * 50)
    print("Example 3: Batch Conversion")
    example_batch_convert()
    print()

