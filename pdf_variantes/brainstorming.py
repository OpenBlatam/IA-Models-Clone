"""
PDF Brainstorming
================

Brainstorming ideas generator from PDF documents.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import random
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class BrainstormIdea:
    """A brainstorm idea."""
    idea: str
    category: str
    related_topics: List[str] = field(default_factory=list)
    potential_impact: str = "medium"
    implementation_difficulty: str = "medium"
    priority_score: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "idea": self.idea,
            "category": self.category,
            "related_topics": self.related_topics,
            "potential_impact": self.potential_impact,
            "implementation_difficulty": self.implementation_difficulty,
            "priority_score": self.priority_score
        }


class PDFBrainstorming:
    """Brainstorming ideas generator from PDF documents."""
    
    def __init__(self, upload_dir: Optional[Path] = None):
        self.upload_dir = upload_dir or Path("./uploads/pdf_variantes")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialized PDF brainstorming")
    
    async def generate_ideas(
        self,
        topics: List[str],
        number_of_ideas: int = 20,
        diversity_level: float = 0.7
    ) -> List[BrainstormIdea]:
        """
        Generate brainstorm ideas from topics.
        
        Args:
            topics: List of topics to brainstorm from
            number_of_ideas: Number of ideas to generate
            diversity_level: Level of diversity in ideas (0-1)
            
        Returns:
            List of brainstorm ideas
        """
        if not topics:
            return []
    
        ideas = []
        categories = [
            "enhancement",
            "improvement",
            "application",
            "expansion",
            "optimization",
            "innovation",
            "analysis",
            "automation",
            "integration",
            "communication"
        ]
        
        # Generate ideas for each topic
        for topic in topics[:10]:  # Limit topics
            topic_ideas = await self._generate_topic_ideas(
                topic,
                categories,
                number_of_ideas // len(topics) if topics else number_of_ideas,
                diversity_level
            )
            ideas.extend(topic_ideas)
        
        # Randomly sample if we have too many
        if len(ideas) > number_of_ideas:
            random.shuffle(ideas)
            ideas = ideas[:number_of_ideas]
        
        # Score ideas
        for idea in ideas:
            idea.priority_score = self._calculate_priority_score(idea)
        
        # Sort by priority
        ideas.sort(key=lambda x: x.priority_score, reverse=True)
        
        logger.info(f"Generated {len(ideas)} brainstorm ideas")
        
        return ideas
    
    async def _generate_topic_ideas(
        self,
        topic: str,
        categories: List[str],
        count: int,
        diversity_level: float
    ) -> List[BrainstormIdea]:
        """Generate ideas for a specific topic."""
        ideas = []
        
        # Select diverse categories
        num_categories = int(len(categories) * diversity_level)
        selected_categories = random.sample(categories, min(num_categories, len(categories)))
        
        for i in range(count):
            category = random.choice(selected_categories)
            
            idea_text = self._generate_idea_text(topic, category)
            
            idea = BrainstormIdea(
                idea=idea_text,
                category=category,
                related_topics=[topic],
                potential_impact=self._randomize_impact(),
                implementation_difficulty=self._randomize_difficulty()
            )
            
            ideas.append(idea)
        
        return ideas
    
    def _generate_idea_text(self, topic: str, category: str) -> str:
        """Generate idea text based on topic and category."""
        
        templates = {
            "enhancement": [
                f"Enhance the {topic} with advanced features",
                f"Improve the capabilities of {topic}",
                f"Upgrade {topic} to the next level"
            ],
            "improvement": [
                f"Improve the efficiency of {topic}",
                f"Optimize {topic} for better performance",
                f"Refine {topic} to meet higher standards"
            ],
            "application": [
                f"Apply {topic} to real-world scenarios",
                f"Use {topic} in new contexts",
                f"Implement {topic} across different domains"
            ],
            "expansion": [
                f"Expand the scope of {topic}",
                f"Extend {topic} to cover more areas",
                f"Broaden the reach of {topic}"
            ],
            "optimization": [
                f"Optimize {topic} for maximum efficiency",
                f"Streamline processes related to {topic}",
                f"Fine-tune {topic} for better results"
            ],
            "innovation": [
                f"Develop innovative solutions for {topic}",
                f"Create new approaches to {topic}",
                f"Design cutting-edge applications of {topic}"
            ],
            "analysis": [
                f"Conduct in-depth analysis of {topic}",
                f"Evaluate the impact of {topic}",
                f"Investigate the implications of {topic}"
            ],
            "automation": [
                f"Automate processes involving {topic}",
                f"Create automated systems for {topic}",
                f"Develop AI-driven solutions for {topic}"
            ],
            "integration": [
                f"Integrate {topic} with existing systems",
                f"Connect {topic} to other platforms",
                f"Unify {topic} with related tools"
            ],
            "communication": [
                f"Improve communication about {topic}",
                f"Create better documentation for {topic}",
                f"Enhance understanding of {topic}"
            ]
        }
        
        category_templates = templates.get(category, templates["enhancement"])
        return random.choice(category_templates)
    
    def _randomize_impact(self) -> str:
        """Randomize potential impact."""
        impacts = ["low", "medium", "high"]
        weights = [0.2, 0.5, 0.3]
        return random.choices(impacts, weights=weights)[0]
    
    def _randomize_difficulty(self) -> str:
        """Randomize implementation difficulty."""
        difficulties = ["easy", "medium", "hard"]
        weights = [0.3, 0.5, 0.2]
        return random.choices(difficulties, weights=weights)[0]
    
    def _calculate_priority_score(self, idea: BrainstormIdea) -> float:
        """Calculate priority score for an idea."""
        score = 0.5  # Base score
        
        # Impact weight
        impact_scores = {"low": 0.3, "medium": 0.5, "high": 0.7}
        score += impact_scores.get(idea.potential_impact, 0.5) * 0.4
        
        # Difficulty weight (easier is better for priority)
        difficulty_scores = {"easy": 0.7, "medium": 0.5, "hard": 0.3}
        score += difficulty_scores.get(idea.implementation_difficulty, 0.5) * 0.3
        
        # Category bonus
        high_value_categories = ["innovation", "optimization", "application"]
        if idea.category in high_value_categories:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    async def organize_by_category(
        self,
        ideas: List[BrainstormIdea]
    ) -> Dict[str, List[BrainstormIdea]]:
        """Organize ideas by category."""
        organized: Dict[str, List[BrainstormIdea]] = {}
        
        for idea in ideas:
            if idea.category not in organized:
                organized[idea.category] = []
            organized[idea.category].append(idea)
        
        # Sort each category by priority
        for category in organized:
            organized[category].sort(key=lambda x: x.priority_score, reverse=True)
        
        return organized
    
    async def filter_by_difficulty(
        self,
        ideas: List[BrainstormIdea],
        difficulty: str
    ) -> List[BrainstormIdea]:
        """Filter ideas by implementation difficulty."""
        return [idea for idea in ideas if idea.implementation_difficulty == difficulty]
    
    async def filter_by_impact(
        self,
        ideas: List[BrainstormIdea],
        impact: str
    ) -> List[BrainstormIdea]:
        """Filter ideas by potential impact."""
        return [idea for idea in ideas if idea.potential_impact == impact]
    
    async def get_top_ideas(
        self,
        ideas: List[BrainstormIdea],
        top_n: int = 10
    ) -> List[BrainstormIdea]:
        """Get top N ideas by priority score."""
        sorted_ideas = sorted(ideas, key=lambda x: x.priority_score, reverse=True)
        return sorted_ideas[:top_n]
    
    async def save_ideas(
        self,
        file_id: str,
        ideas: List[BrainstormIdea]
    ) -> str:
        """Save brainstorm ideas to file."""
        ideas_file = self.upload_dir / f"{file_id}_brainstorm.json"
        
        data = {
            "file_id": file_id,
            "ideas": [idea.to_dict() for idea in ideas],
            "generated_at": datetime.utcnow().isoformat(),
            "total_ideas": len(ideas)
        }
        
        with open(ideas_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(ideas)} brainstorm ideas for {file_id}")
        
        return ideas_file
    
    async def load_ideas(self, file_id: str) -> List[BrainstormIdea]:
        """Load brainstorm ideas from file."""
        ideas_file = self.upload_dir / f"{file_id}_brainstorm.json"
        
        if not ideas_file.exists():
            return []
        
        try:
            with open(ideas_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                ideas = []
                for idea_data in data.get("ideas", []):
                    ideas.append(BrainstormIdea(
                        idea=idea_data["idea"],
                        category=idea_data["category"],
                        related_topics=idea_data.get("related_topics", []),
                        potential_impact=idea_data.get("potential_impact", "medium"),
                        implementation_difficulty=idea_data.get("implementation_difficulty", "medium"),
                        priority_score=idea_data.get("priority_score", 0.5)
                    ))
                
                return ideas
                
        except Exception as e:
            logger.error(f"Error loading ideas: {e}")
            return []
