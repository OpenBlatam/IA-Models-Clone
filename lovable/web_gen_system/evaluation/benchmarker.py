from typing import Dict, Any, List

class WebGenBench:
    """
    Evaluates the generated web project based on functional and visual metrics
    (WebGen-Bench concept).
    """
    
    def evaluate_project(self, code_structure: Dict[str, str], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates a score for the generated project.
        """
        metrics = {
            "file_structure_score": 0.0,
            "content_relevance_score": 0.0,
            "visual_heuristic_score": 0.0,
            "total_score": 0.0
        }
        
        if not code_structure:
            return metrics
            
        # 1. File Structure Score (Do we have the basics?)
        required_files = ["package.json"]
        if any("app" in p for p in code_structure.keys()):
             required_files.append("app") # Next.js / Expo convention
             
        found_files = 0
        for req in required_files:
            if any(req in path for path in code_structure.keys()):
                found_files += 1
        
        metrics["file_structure_score"] = (found_files / len(required_files)) * 100
        
        # 2. Content Relevance Score (Did we implement requested features?)
        core_features = requirements.get("core_features", [])
        if core_features:
            features_found = 0
            all_content = " ".join(code_structure.values()).lower()
            
            for feature in core_features:
                # Naive keyword matching
                keywords = feature.lower().split()
                # Check if at least one significant keyword exists
                if any(kw in all_content for kw in keywords if len(kw) > 3):
                    features_found += 1
            
            metrics["content_relevance_score"] = (features_found / len(core_features)) * 100
        else:
            metrics["content_relevance_score"] = 100.0 # No specific features requested
            
        # 3. Visual Heuristic Score (Penalize for obvious issues)
        # We re-use some logic from VisualCritic or just do simple checks here
        penalty = 0
        for path, content in code_structure.items():
            if "<img" in content and "alt=" not in content:
                penalty += 5
            if "TODO" in content:
                penalty += 2
                
        metrics["visual_heuristic_score"] = max(0.0, 100.0 - penalty)
        
        # Total Score (Weighted Average)
        metrics["total_score"] = (
            metrics["file_structure_score"] * 0.3 +
            metrics["content_relevance_score"] * 0.4 +
            metrics["visual_heuristic_score"] * 0.3
        )
        
        return metrics
