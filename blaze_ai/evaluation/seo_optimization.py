"""
SEO optimization evaluation metrics.

This module provides comprehensive evaluation metrics for SEO optimization
including keyword density, readability, content structure, and technical SEO metrics.
"""

from __future__ import annotations

import re
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import Counter, defaultdict
import string

import numpy as np
from bs4 import BeautifulSoup
import requests


def calculate_keyword_density(text: str, keywords: List[str]) -> Dict[str, float]:
    """Calculate keyword density for given keywords in text."""
    # Clean and normalize text
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    total_words = len(words)
    
    if total_words == 0:
        return {keyword: 0.0 for keyword in keywords}
    
    # Calculate density for each keyword
    densities = {}
    for keyword in keywords:
        keyword_lower = keyword.lower()
        # Count exact matches
        exact_matches = len(re.findall(rf'\b{re.escape(keyword_lower)}\b', text_lower))
        # Count partial matches (for multi-word keywords)
        if ' ' in keyword_lower:
            partial_matches = len(re.findall(rf'\b{re.escape(keyword_lower)}\b', text_lower))
        else:
            partial_matches = exact_matches
        
        density = (partial_matches / total_words) * 100
        densities[keyword] = round(density, 4)
    
    return densities


def calculate_readability_metrics(text: str) -> Dict[str, float]:
    """Calculate various readability metrics."""
    metrics = {}
    
    # Basic text statistics
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    words = re.findall(r'\b\w+\b', text)
    syllables = sum(len(re.findall(r'[aeiouy]+', word.lower())) for word in words)
    
    # Count characters (excluding spaces)
    characters = len(re.sub(r'\s', '', text))
    
    # Flesch Reading Ease
    if len(words) > 0 and len(sentences) > 0:
        flesch_score = 206.835 - (1.015 * (len(words) / len(sentences))) - (84.6 * (syllables / len(words)))
        flesch_score = max(0, min(100, flesch_score))
        metrics["flesch_reading_ease"] = round(flesch_score, 2)
        
        # Flesch-Kincaid Grade Level
        fk_grade = 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
        metrics["flesch_kincaid_grade"] = round(max(0, fk_grade), 1)
    else:
        metrics["flesch_reading_ease"] = 0.0
        metrics["flesch_kincaid_grade"] = 0.0
    
    # Gunning Fog Index
    if len(words) > 0 and len(sentences) > 0:
        complex_words = sum(1 for word in words if len(re.findall(r'[aeiouy]+', word.lower())) > 2)
        fog_index = 0.4 * ((len(words) / len(sentences)) + 100 * (complex_words / len(words)))
        metrics["gunning_fog_index"] = round(fog_index, 1)
    else:
        metrics["gunning_fog_index"] = 0.0
    
    # SMOG Index
    if len(sentences) > 0:
        complex_words = sum(1 for word in words if len(re.findall(r'[aeiouy]+', word.lower())) > 2)
        smog_index = 1.043 * math.sqrt(complex_words * (30 / len(sentences))) + 3.1291
        metrics["smog_index"] = round(smog_index, 1)
    else:
        metrics["smog_index"] = 0.0
    
    # Basic statistics
    metrics["total_sentences"] = len(sentences)
    metrics["total_words"] = len(words)
    metrics["total_syllables"] = syllables
    metrics["total_characters"] = characters
    metrics["avg_words_per_sentence"] = round(len(words) / max(len(sentences), 1), 2)
    metrics["avg_syllables_per_word"] = round(syllables / max(len(words), 1), 2)
    
    return metrics


def analyze_content_structure(text: str) -> Dict[str, Union[int, float, List[str]]]:
    """Analyze content structure and organization."""
    analysis = {}
    
    # Paragraph analysis
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    analysis["total_paragraphs"] = len(paragraphs)
    
    # Heading analysis
    heading_patterns = [
        (r'^#{1,6}\s+(.+)$', 'markdown_h1_h6'),
        (r'^<h[1-6]>(.+?)</h[1-6]>$', 'html_h1_h6'),
        (r'^(.+?)\n=+\n$', 'markdown_h1'),
        (r'^(.+?)\n-+\n$', 'markdown_h2')
    ]
    
    headings = []
    for pattern, heading_type in heading_patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        headings.extend([(match.strip(), heading_type) for match in matches])
    
    analysis["total_headings"] = len(headings)
    analysis["headings"] = headings
    
    # List analysis
    list_items = re.findall(r'^[\s]*[-*+]\s+(.+)$', text, re.MULTILINE)
    numbered_list_items = re.findall(r'^[\s]*\d+\.\s+(.+)$', text, re.MULTILINE)
    
    analysis["total_list_items"] = len(list_items) + len(numbered_list_items)
    analysis["bullet_list_items"] = len(list_items)
    analysis["numbered_list_items"] = len(numbered_list_items)
    
    # Link analysis
    links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', text)  # Markdown links
    html_links = re.findall(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>', text)
    
    analysis["total_links"] = len(links) + len(html_links)
    analysis["markdown_links"] = links
    analysis["html_links"] = html_links
    
    # Image analysis
    images = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', text)  # Markdown images
    html_images = re.findall(r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>', text)
    
    analysis["total_images"] = len(images) + len(html_images)
    analysis["markdown_images"] = images
    analysis["html_images"] = html_images
    
    # Content distribution
    if paragraphs:
        paragraph_lengths = [len(p.split()) for p in paragraphs]
        analysis["avg_paragraph_length"] = round(np.mean(paragraph_lengths), 2)
        analysis["min_paragraph_length"] = min(paragraph_lengths)
        analysis["max_paragraph_length"] = max(paragraph_lengths)
    else:
        analysis["avg_paragraph_length"] = 0.0
        analysis["min_paragraph_length"] = 0
        analysis["max_paragraph_length"] = 0
    
    return analysis


def calculate_seo_technical_metrics(text: str, url: Optional[str] = None) -> Dict[str, Union[int, float, str]]:
    """Calculate technical SEO metrics."""
    metrics = {}
    
    # Meta tag analysis (if HTML)
    if '<html' in text.lower() or '<head' in text.lower():
        soup = BeautifulSoup(text, 'html.parser')
        
        # Title tag
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
            metrics["title_length"] = len(title)
            metrics["title"] = title
            metrics["title_optimal"] = 50 <= len(title) <= 60
        else:
            metrics["title_length"] = 0
            metrics["title"] = ""
            metrics["title_optimal"] = False
        
        # Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            description = meta_desc.get('content', '').strip()
            metrics["meta_description_length"] = len(description)
            metrics["meta_description"] = description
            metrics["meta_description_optimal"] = 150 <= len(description) <= 160
        else:
            metrics["meta_description_length"] = 0
            metrics["meta_description"] = ""
            metrics["meta_description_optimal"] = False
        
        # H1 tags
        h1_tags = soup.find_all('h1')
        metrics["h1_count"] = len(h1_tags)
        metrics["h1_optimal"] = len(h1_tags) == 1
        
        # H2-H6 tags
        h2_tags = soup.find_all('h2')
        h3_tags = soup.find_all('h3')
        metrics["h2_count"] = len(h2_tags)
        metrics["h3_count"] = len(h3_tags)
        
        # Alt text for images
        images = soup.find_all('img')
        images_with_alt = [img for img in images if img.get('alt')]
        metrics["total_images"] = len(images)
        metrics["images_with_alt"] = len(images_with_alt)
        metrics["alt_text_coverage"] = round(len(images_with_alt) / max(len(images), 1) * 100, 2)
        
        # Internal vs external links
        internal_links = []
        external_links = []
        
        if url:
            domain = re.search(r'https?://([^/]+)', url)
            if domain:
                domain = domain.group(1)
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.startswith('/') or domain in href:
                        internal_links.append(href)
                    else:
                        external_links.append(href)
        
        metrics["internal_links"] = len(internal_links)
        metrics["external_links"] = len(external_links)
        metrics["total_links"] = len(internal_links) + len(external_links)
    
    # Text-based metrics
    words = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words)
    
    # Content length
    metrics["word_count"] = total_words
    metrics["character_count"] = len(text)
    metrics["content_optimal"] = 300 <= total_words <= 2500
    
    # Keyword optimization
    if total_words > 0:
        # Check for keyword stuffing
        word_freq = Counter(words)
        most_common_word = word_freq.most_common(1)[0]
        keyword_density = (most_common_word[1] / total_words) * 100
        metrics["keyword_density_max"] = round(keyword_density, 2)
        metrics["keyword_stuffing_risk"] = keyword_density > 5.0  # 5% threshold
    else:
        metrics["word_count"] = 0
        metrics["character_count"] = 0
        metrics["content_optimal"] = False
        metrics["keyword_density_max"] = 0.0
        metrics["keyword_stuffing_risk"] = False
    
    return metrics


def calculate_content_quality_score(text: str, target_keywords: List[str]) -> Dict[str, Union[float, str]]:
    """Calculate overall content quality score."""
    score_components = {}
    total_score = 0
    max_possible_score = 100
    
    # Readability score (25 points)
    readability = calculate_readability_metrics(text)
    flesch_score = readability.get("flesch_reading_ease", 0)
    
    if flesch_score >= 80:
        readability_score = 25
    elif flesch_score >= 60:
        readability_score = 20
    elif flesch_score >= 40:
        readability_score = 15
    elif flesch_score >= 20:
        readability_score = 10
    else:
        readability_score = 5
    
    score_components["readability_score"] = readability_score
    total_score += readability_score
    
    # Content length score (20 points)
    word_count = readability.get("total_words", 0)
    if 300 <= word_count <= 2500:
        length_score = 20
    elif 200 <= word_count < 300 or 2500 < word_count <= 3000:
        length_score = 15
    elif 100 <= word_count < 200 or 3000 < word_count <= 4000:
        length_score = 10
    else:
        length_score = 5
    
    score_components["length_score"] = length_score
    total_score += length_score
    
    # Keyword optimization score (25 points)
    if target_keywords:
        keyword_densities = calculate_keyword_density(text, target_keywords)
        keyword_scores = []
        
        for keyword, density in keyword_densities.items():
            if 0.5 <= density <= 2.5:  # Optimal range
                keyword_scores.append(25 / len(target_keywords))
            elif 0.1 <= density <= 5.0:  # Acceptable range
                keyword_scores.append(15 / len(target_keywords))
            elif density > 0:
                keyword_scores.append(5 / len(target_keywords))
            else:
                keyword_scores.append(0)
        
        keyword_score = sum(keyword_scores)
    else:
        keyword_score = 0
    
    score_components["keyword_score"] = keyword_score
    total_score += keyword_score
    
    # Content structure score (20 points)
    structure = analyze_content_structure(text)
    
    structure_score = 0
    if structure.get("total_headings", 0) > 0:
        structure_score += 5
    if structure.get("total_paragraphs", 0) >= 3:
        structure_score += 5
    if structure.get("total_list_items", 0) > 0:
        structure_score += 5
    if structure.get("total_images", 0) > 0:
        structure_score += 5
    
    score_components["structure_score"] = structure_score
    total_score += structure_score
    
    # Technical SEO score (10 points)
    technical = calculate_seo_technical_metrics(text)
    
    technical_score = 0
    if technical.get("title_optimal", False):
        technical_score += 3
    if technical.get("meta_description_optimal", False):
        technical_score += 3
    if technical.get("h1_optimal", False):
        technical_score += 2
    if technical.get("alt_text_coverage", 0) >= 80:
        technical_score += 2
    
    score_components["technical_score"] = technical_score
    total_score += technical_score
    
    # Overall score and grade
    overall_score = round(total_score, 1)
    score_components["overall_score"] = overall_score
    
    if overall_score >= 90:
        grade = "A+"
    elif overall_score >= 80:
        grade = "A"
    elif overall_score >= 70:
        grade = "B"
    elif overall_score >= 60:
        grade = "C"
    elif overall_score >= 50:
        grade = "D"
    else:
        grade = "F"
    
    score_components["grade"] = grade
    score_components["max_possible_score"] = max_possible_score
    
    return score_components


def evaluate_seo_optimization(text: str, target_keywords: List[str], 
                            url: Optional[str] = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation of SEO optimization.
    
    Args:
        text: The text content to evaluate
        target_keywords: List of target keywords to optimize for
        url: Optional URL for technical SEO analysis
    
    Returns:
        Dictionary containing all SEO evaluation metrics
    """
    evaluation = {}
    
    # Basic SEO metrics
    evaluation["keyword_density"] = calculate_keyword_density(text, target_keywords)
    
    # Readability analysis
    evaluation["readability"] = calculate_readability_metrics(text)
    
    # Content structure analysis
    evaluation["content_structure"] = analyze_content_structure(text)
    
    # Technical SEO metrics
    evaluation["technical_seo"] = calculate_seo_technical_metrics(text, url)
    
    # Overall quality score
    evaluation["quality_score"] = calculate_content_quality_score(text, target_keywords)
    
    # Recommendations
    recommendations = []
    
    # Readability recommendations
    flesch_score = evaluation["readability"].get("flesch_reading_ease", 0)
    if flesch_score < 60:
        recommendations.append("Improve readability by using shorter sentences and simpler words")
    
    # Content length recommendations
    word_count = evaluation["readability"].get("total_words", 0)
    if word_count < 300:
        recommendations.append("Increase content length to at least 300 words for better SEO")
    elif word_count > 2500:
        recommendations.append("Consider breaking long content into multiple pages")
    
    # Keyword optimization recommendations
    for keyword, density in evaluation["keyword_density"].items():
        if density < 0.5:
            recommendations.append(f"Increase usage of keyword '{keyword}' (current: {density}%)")
        elif density > 2.5:
            recommendations.append(f"Reduce overuse of keyword '{keyword}' (current: {density}%)")
    
    # Structure recommendations
    structure = evaluation["content_structure"]
    if structure.get("total_headings", 0) == 0:
        recommendations.append("Add headings to improve content structure and readability")
    if structure.get("total_paragraphs", 0) < 3:
        recommendations.append("Break content into more paragraphs for better readability")
    
    # Technical recommendations
    technical = evaluation["technical_seo"]
    if not technical.get("title_optimal", False):
        recommendations.append("Optimize title tag length (50-60 characters)")
    if not technical.get("meta_description_optimal", False):
        recommendations.append("Optimize meta description length (150-160 characters)")
    if not technical.get("h1_optimal", False):
        recommendations.append("Ensure exactly one H1 tag per page")
    
    evaluation["recommendations"] = recommendations
    
    return evaluation
