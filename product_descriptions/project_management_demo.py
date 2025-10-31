from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import os
import sys
import time
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from project_management_system import (
from fastapi import FastAPI, HTTPException
from project_management_system import ProjectManager, create_sample_problem_definition
import gradio as gr
from project_management_system import ProjectManager
            import traceback
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive Project Management System Demo

This demo showcases the complete project management system with:

1. Problem Definition Creation
2. Dataset Analysis and Validation
3. Task Management and Tracking
4. Project Progress Monitoring
5. Comprehensive Reporting
6. Visualization and Analytics
7. Team Collaboration Features
8. Integration Examples
"""


# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    ProjectManager, ProblemDefinition, Task, TaskStatus, 
    ProjectStatus, ProblemType, DatasetType, create_sample_problem_definition
)

# Configure warnings and plotting
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ProjectManagementDemo:
    """Comprehensive demo for the project management system."""
    
    def __init__(self) -> Any:
        self.manager = ProjectManager("./demo_projects")
        self.demo_data = None
        self.setup_demo_data()
    
    def setup_demo_data(self) -> Any:
        """Create sample dataset for demonstration."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create synthetic customer churn dataset
        data = {
            'customer_id': range(1, n_samples + 1),
            'age': np.random.normal(45, 15, n_samples).astype(int),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'tenure': np.random.exponential(5, n_samples).astype(int),
            'monthly_charges': np.random.normal(65, 20, n_samples),
            'total_charges': np.random.normal(2000, 1000, n_samples),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
            'monthly_usage_gb': np.random.exponential(50, n_samples)
        }
        
        # Create target variable with some logic
        churn_prob = (
            (data['tenure'] < 12) * 0.3 +
            (data['monthly_charges'] > 80) * 0.2 +
            (data['contract_type'] == 'Month-to-month') * 0.25 +
            (data['payment_method'] == 'Electronic check') * 0.15 +
            np.random.normal(0, 0.1, n_samples)
        )
        data['churn'] = (churn_prob > 0.5).astype(int)
        
        # Add some missing values
        missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        data['monthly_usage_gb'][missing_indices] = np.nan
        
        self.demo_data = pd.DataFrame(data)
    
    def demo_problem_definition(self) -> Any:
        """Demonstrate problem definition creation."""
        print("=" * 80)
        print("PROBLEM DEFINITION DEMONSTRATION")
        print("=" * 80)
        
        # Create sample problem definition
        problem_def = create_sample_problem_definition()
        
        print(f"Problem Title: {problem_def.title}")
        print(f"Problem Type: {problem_def.problem_type.value}")
        print(f"Business Objective: {problem_def.business_objective}")
        print(f"Timeline: {problem_def.timeline_weeks} weeks")
        print(f"Team Size: {problem_def.team_size} members")
        print(f"Budget: ${problem_def.budget:,.2f}")
        
        print("\nSuccess Criteria:")
        for i, criterion in enumerate(problem_def.success_criteria, 1):
            print(f"  {i}. {criterion}")
        
        print("\nEvaluation Metrics:")
        for metric in problem_def.evaluation_metrics:
            print(f"  - {metric}")
        
        print("\nConstraints:")
        for constraint in problem_def.constraints:
            print(f"  - {constraint}")
        
        print("\nRisks:")
        for risk in problem_def.risks:
            print(f"  - {risk}")
        
        return problem_def
    
    def demo_project_creation(self, problem_def: ProblemDefinition):
        """Demonstrate project creation."""
        print("\n" + "=" * 80)
        print("PROJECT CREATION DEMONSTRATION")
        print("=" * 80)
        
        # Create project
        project = self.manager.create_project(
            project_id="churn_prediction_001",
            name="Customer Churn Prediction",
            problem_definition=problem_def
        )
        
        print(f"Project Created Successfully!")
        print(f"Project ID: {project.id}")
        print(f"Project Name: {project.name}")
        print(f"Status: {project.status.value}")
        print(f"Progress: {project.progress_percentage}%")
        print(f"Created Date: {project.created_date}")
        
        return project
    
    def demo_dataset_analysis(self, project_id: str):
        """Demonstrate dataset analysis."""
        print("\n" + "=" * 80)
        print("DATASET ANALYSIS DEMONSTRATION")
        print("=" * 80)
        
        # Add dataset to project
        dataset_info = self.manager.add_dataset(
            project_id=project_id,
            data=self.demo_data,
            dataset_name="customer_churn_data"
        )
        
        print(f"Dataset Added: {dataset_info.name}")
        print(f"Dataset Type: {dataset_info.dataset_type.value}")
        print(f"Samples: {dataset_info.num_samples:,}")
        print(f"Features: {dataset_info.num_features}")
        print(f"Missing Values: {sum(dataset_info.missing_values.values())}")
        print(f"Duplicate Rows: {dataset_info.duplicate_rows}")
        
        # Create data quality report
        quality_report = self.manager.dataset_analyzer.create_data_quality_report(dataset_info)
        
        print("\nData Quality Report:")
        print(f"  Missing Percentage: {quality_report['data_quality']['missing_percentage']:.2f}%")
        print(f"  Duplicate Percentage: {quality_report['data_quality']['duplicate_percentage']:.2f}%")
        print(f"  Numerical Features: {quality_report['feature_analysis']['numerical_features']}")
        print(f"  Categorical Features: {quality_report['feature_analysis']['categorical_features']}")
        
        # Show statistical summary
        print("\nNumerical Features Summary:")
        for feature, stats in dataset_info.numerical_stats.items():
            print(f"  {feature}:")
            print(f"    Mean: {stats['mean']:.2f}")
            print(f"    Std: {stats['std']:.2f}")
            print(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        
        return dataset_info
    
    def demo_task_management(self, project_id: str):
        """Demonstrate task management."""
        print("\n" + "=" * 80)
        print("TASK MANAGEMENT DEMONSTRATION")
        print("=" * 80)
        
        # Create sample tasks
        tasks = [
            Task(
                id="task_001",
                title="Data Exploration and Analysis",
                description="Perform comprehensive exploratory data analysis on customer churn dataset",
                assigned_to="Data Scientist",
                priority=1,
                estimated_hours=8.0,
                tags=["data-analysis", "exploration"]
            ),
            Task(
                id="task_002",
                title="Feature Engineering",
                description="Create new features and transform existing ones for better model performance",
                assigned_to="ML Engineer",
                priority=1,
                estimated_hours=12.0,
                dependencies=["task_001"],
                tags=["feature-engineering", "preprocessing"]
            ),
            Task(
                id="task_003",
                title="Model Development",
                description="Develop and train machine learning models for churn prediction",
                assigned_to="ML Engineer",
                priority=1,
                estimated_hours=16.0,
                dependencies=["task_002"],
                tags=["modeling", "training"]
            ),
            Task(
                id="task_004",
                title="Model Evaluation",
                description="Evaluate model performance using various metrics and validation techniques",
                assigned_to="Data Scientist",
                priority=2,
                estimated_hours=6.0,
                dependencies=["task_003"],
                tags=["evaluation", "validation"]
            ),
            Task(
                id="task_005",
                title="Model Deployment",
                description="Deploy the best performing model to production environment",
                assigned_to="DevOps Engineer",
                priority=2,
                estimated_hours=10.0,
                dependencies=["task_004"],
                tags=["deployment", "production"]
            ),
            Task(
                id="task_006",
                title="Documentation",
                description="Create comprehensive documentation for the project",
                assigned_to="Technical Writer",
                priority=3,
                estimated_hours=4.0,
                dependencies=["task_005"],
                tags=["documentation"]
            )
        ]
        
        # Add tasks to project
        for task in tasks:
            self.manager.add_task(project_id, task)
            print(f"Task Added: {task.title} (Priority: {task.priority})")
        
        # Update task statuses to simulate progress
        print("\nUpdating Task Statuses...")
        
        # Complete data exploration
        self.manager.update_task_status(project_id, "task_001", TaskStatus.COMPLETED)
        print("  ✓ Data Exploration completed")
        
        # Start feature engineering
        self.manager.update_task_status(project_id, "task_002", TaskStatus.IN_PROGRESS, 75.0)
        print("  ⟳ Feature Engineering in progress (75%)")
        
        # Start model development
        self.manager.update_task_status(project_id, "task_003", TaskStatus.IN_PROGRESS, 25.0)
        print("  ⟳ Model Development in progress (25%)")
        
        # Block model evaluation (waiting for model development)
        self.manager.update_task_status(project_id, "task_004", TaskStatus.BLOCKED)
        print("  ⚠ Model Evaluation blocked (waiting for dependencies)")
        
        return tasks
    
    def demo_project_summary(self, project_id: str):
        """Demonstrate project summary and reporting."""
        print("\n" + "=" * 80)
        print("PROJECT SUMMARY AND REPORTING")
        print("=" * 80)
        
        # Get project summary
        summary = self.manager.get_project_summary(project_id)
        
        print("Project Information:")
        print(f"  Name: {summary['project_info']['name']}")
        print(f"  Status: {summary['project_info']['status']}")
        print(f"  Progress: {summary['project_info']['progress_percentage']:.1f}%")
        print(f"  Created: {summary['project_info']['created_date']}")
        print(f"  Last Updated: {summary['project_info']['last_updated']}")
        
        print("\nProblem Definition:")
        print(f"  Title: {summary['problem_definition']['title']}")
        print(f"  Type: {summary['problem_definition']['problem_type']}")
        print(f"  Objective: {summary['problem_definition']['business_objective']}")
        
        print("\nTask Statistics:")
        task_stats = summary['task_statistics']
        print(f"  Total Tasks: {task_stats['total']}")
        print(f"  Completed: {task_stats['completed']}")
        print(f"  In Progress: {task_stats['in_progress']}")
        print(f"  Todo: {task_stats['todo']}")
        print(f"  Blocked: {task_stats['blocked']}")
        
        print("\nDataset Statistics:")
        dataset_stats = summary['dataset_statistics']
        print(f"  Total Datasets: {dataset_stats['total_datasets']}")
        print(f"  Total Samples: {dataset_stats['total_samples']:,}")
        print(f"  Total Features: {dataset_stats['total_features']}")
        
        print("\nTeam Information:")
        team_info = summary['team_info']
        print(f"  Team Size: {team_info['team_size']}")
        print(f"  Team Members: {', '.join(team_info['team_members']) if team_info['team_members'] else 'Not assigned'}")
    
    def demo_comprehensive_report(self, project_id: str):
        """Demonstrate comprehensive project report."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE PROJECT REPORT")
        print("=" * 80)
        
        # Create comprehensive report
        report = self.manager.create_project_report(project_id)
        
        print("Report Generated Successfully!")
        print(f"Generated Date: {report['generated_date']}")
        
        # Task analysis
        task_analysis = report['task_analysis']
        
        print("\nTask Analysis by Status:")
        for status, count in task_analysis['by_status'].items():
            print(f"  {status}: {count}")
        
        print("\nTask Analysis by Priority:")
        for priority, count in task_analysis['by_priority'].items():
            print(f"  {priority}: {count}")
        
        print("\nTask Analysis by Assignment:")
        for assignee, count in task_analysis['by_assignment'].items():
            print(f"  {assignee}: {count}")
        
        # Recommendations
        print("\nRecommendations:")
        for i, recommendation in enumerate(report['recommendations'], 1):
            print(f"  {i}. {recommendation}")
        
        # Dataset reports
        print(f"\nDataset Reports: {len(report['dataset_reports'])}")
        for i, dataset_report in enumerate(report['dataset_reports'], 1):
            overview = dataset_report['dataset_overview']
            quality = dataset_report['data_quality']
            print(f"  Dataset {i}: {overview['name']}")
            print(f"    Type: {overview['type']}")
            print(f"    Samples: {overview['samples']:,}")
            print(f"    Missing: {quality['missing_percentage']:.2f}%")
            print(f"    Duplicates: {quality['duplicate_percentage']:.2f}%")
    
    def demo_visualization(self, project_id: str):
        """Demonstrate data visualization capabilities."""
        print("\n" + "=" * 80)
        print("DATA VISUALIZATION DEMONSTRATION")
        print("=" * 80)
        
        # Create visualization report
        figures = self.manager.dataset_analyzer.create_visualization_report(
            self.demo_data, 
            self.manager.projects[project_id].datasets[0]
        )
        
        print(f"Generated {len(figures)} visualization figures:")
        for figure_name in figures.keys():
            print(f"  - {figure_name}")
        
        # Save figures
        output_dir = Path("./demo_visualizations")
        output_dir.mkdir(exist_ok=True)
        
        for figure_name, fig in figures.items():
            fig.write_html(output_dir / f"{figure_name}.html")
            print(f"  Saved: {figure_name}.html")
        
        print(f"\nVisualizations saved to: {output_dir.absolute()}")
    
    def demo_advanced_features(self, project_id: str):
        """Demonstrate advanced project management features."""
        print("\n" + "=" * 80)
        print("ADVANCED FEATURES DEMONSTRATION")
        print("=" * 80)
        
        # Simulate project timeline
        project = self.manager.projects[project_id]
        project.start_date = datetime.now() - timedelta(days=7)
        project.end_date = datetime.now() + timedelta(days=49)  # 8 weeks
        project.team_members = ["Alice (Data Scientist)", "Bob (ML Engineer)", "Charlie (DevOps)", "Diana (Product Manager)"]
        
        # Add milestones
        project.milestones = [
            {
                "id": "milestone_001",
                "title": "Data Analysis Complete",
                "description": "Exploratory data analysis and data quality assessment completed",
                "due_date": (datetime.now() + timedelta(days=7)).isoformat(),
                "status": "completed"
            },
            {
                "id": "milestone_002",
                "title": "Model Development Complete",
                "description": "Initial model development and training completed",
                "due_date": (datetime.now() + timedelta(days=28)).isoformat(),
                "status": "in_progress"
            },
            {
                "id": "milestone_003",
                "title": "Model Deployment",
                "description": "Model deployed to production environment",
                "due_date": (datetime.now() + timedelta(days=42)).isoformat(),
                "status": "pending"
            },
            {
                "id": "milestone_004",
                "title": "Project Completion",
                "description": "All deliverables completed and project closed",
                "due_date": (datetime.now() + timedelta(days=49)).isoformat(),
                "status": "pending"
            }
        ]
        
        # Save updated project
        self.manager._save_project(project)
        
        print("Advanced Features Configured:")
        print(f"  Project Timeline: {project.start_date.date()} to {project.end_date.date()}")
        print(f"  Team Members: {len(project.team_members)}")
        print(f"  Milestones: {len(project.milestones)}")
        
        # Show timeline analysis
        summary = self.manager.get_project_summary(project_id)
        timeline_stats = summary['timeline_statistics']
        
        print(f"\nTimeline Analysis:")
        print(f"  Duration: {timeline_stats['duration_days']} days")
        print(f"  Days Remaining: {timeline_stats['days_remaining']}")
        print(f"  Timeline Progress: {timeline_stats['timeline_progress']:.1f}%")
        
        # Show milestones
        print(f"\nMilestones:")
        for milestone in project.milestones:
            status_icon = {
                "completed": "✓",
                "in_progress": "⟳",
                "pending": "⏳"
            }.get(milestone['status'], "?")
            print(f"  {status_icon} {milestone['title']} ({milestone['status']})")
    
    def demo_integration_examples(self) -> Any:
        """Demonstrate integration with other systems."""
        print("\n" + "=" * 80)
        print("INTEGRATION EXAMPLES")
        print("=" * 80)
        
        # Example 1: FastAPI Integration
        print("1. FastAPI Integration Example:")
        print("""

app = FastAPI()
manager = ProjectManager()

@app.post("/projects/")
async def create_project(project_data: dict):
    
    """create_project function."""
try:
        problem_def = create_sample_problem_definition()
        project = manager.create_project(
            project_id=project_data["id"],
            name=project_data["name"],
            problem_definition=problem_def
        )
        return {"status": "success", "project_id": project.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/projects/{project_id}/summary")
async def get_project_summary(project_id: str):
    
    """get_project_summary function."""
try:
        summary = manager.get_project_summary(project_id)
        return summary
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
        """)
        
        # Example 2: Gradio Integration
        print("\n2. Gradio Integration Example:")
        print("""

def create_project_interface(project_name, problem_type, description) -> Any:
    manager = ProjectManager()
    problem_def = create_sample_problem_definition()
    problem_def.title = project_name
    problem_def.description = description
    
    project = manager.create_project(
        project_id=f"project_{int(time.time())}",
        name=project_name,
        problem_definition=problem_def
    )
    
    return f"Project created: {project.id}"

interface = gr.Interface(
    fn=create_project_interface,
    inputs=[
        gr.Textbox(label="Project Name"),
        gr.Dropdown(choices=["classification", "regression"], label="Problem Type"),
        gr.Textbox(label="Description", lines=3)
    ],
    outputs=gr.Textbox(label="Result")
)
        """)
        
        # Example 3: Automated Workflow
        print("\n3. Automated Workflow Example:")
        print("""
async def automated_project_workflow():
    
    """automated_project_workflow function."""
manager = ProjectManager()
    
    # Create project
    problem_def = create_sample_problem_definition()
    project = manager.create_project("auto_001", "Automated Project", problem_def)
    
    # Add dataset
    dataset_info = manager.add_dataset(project.id, data, "auto_dataset")
    
    # Create tasks automatically
    tasks = [
        Task(id="auto_001", title="Auto Task 1", description="Automated task"),
        Task(id="auto_002", title="Auto Task 2", description="Automated task")
    ]
    
    for task in tasks:
        manager.add_task(project.id, task)
    
    return project.id
        """)
    
    def run_complete_demo(self) -> Any:
        """Run the complete demonstration."""
        print("🚀 COMPREHENSIVE PROJECT MANAGEMENT SYSTEM DEMO")
        print("=" * 80)
        
        try:
            # 1. Problem Definition
            problem_def = self.demo_problem_definition()
            
            # 2. Project Creation
            project = self.demo_project_creation(problem_def)
            
            # 3. Dataset Analysis
            dataset_info = self.demo_dataset_analysis(project.id)
            
            # 4. Task Management
            tasks = self.demo_task_management(project.id)
            
            # 5. Project Summary
            self.demo_project_summary(project.id)
            
            # 6. Comprehensive Report
            self.demo_comprehensive_report(project.id)
            
            # 7. Visualization
            self.demo_visualization(project.id)
            
            # 8. Advanced Features
            self.demo_advanced_features(project.id)
            
            # 9. Integration Examples
            self.demo_integration_examples()
            
            print("\n" + "=" * 80)
            print("✅ DEMO COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("\nKey Features Demonstrated:")
            print("  ✓ Problem Definition and Scope Management")
            print("  ✓ Comprehensive Dataset Analysis")
            print("  ✓ Task Management and Progress Tracking")
            print("  ✓ Project Reporting and Analytics")
            print("  ✓ Data Visualization")
            print("  ✓ Timeline and Milestone Management")
            print("  ✓ Team Collaboration Features")
            print("  ✓ Integration Examples")
            
            print(f"\nProject files saved to: {self.manager.project_dir.absolute()}")
            print("Visualizations saved to: ./demo_visualizations/")
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            traceback.print_exc()


def main():
    """Main function to run the demo."""
    demo = ProjectManagementDemo()
    demo.run_complete_demo()


match __name__:
    case "__main__":
    main() 