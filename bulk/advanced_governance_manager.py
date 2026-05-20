"""
BUL Advanced Governance Manager
==============================

Advanced governance management system for the BUL system.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import sqlite3
from dataclasses import dataclass
from enum import Enum
import yaml
from collections import defaultdict

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config_optimized import BULConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GovernanceLevel(Enum):
    """Governance levels."""
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    OPERATIONAL = "operational"

class GovernanceStatus(Enum):
    """Governance status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    ARCHIVED = "archived"

class GovernanceType(Enum):
    """Governance types."""
    POLICY = "policy"
    PROCEDURE = "procedure"
    STANDARD = "standard"
    GUIDELINE = "guideline"
    FRAMEWORK = "framework"

@dataclass
class GovernanceDocument:
    """Governance document definition."""
    id: str
    title: str
    description: str
    governance_type: GovernanceType
    level: GovernanceLevel
    category: str
    owner: str
    approver: str
    version: str
    status: GovernanceStatus
    effective_date: datetime
    review_date: datetime
    content: str
    tags: List[str]
    dependencies: List[str]
    created_at: datetime = None

@dataclass
class GovernanceReview:
    """Governance review definition."""
    id: str
    document_id: str
    reviewer: str
    review_date: datetime
    status: str
    comments: str
    recommendations: List[str]
    next_review: datetime
    approved: bool = False

class AdvancedGovernanceManager:
    """Advanced governance management system for BUL system."""
    
    def __init__(self, config: Optional[BULConfig] = None):
        self.config = config or BULConfig()
        self.documents = {}
        self.reviews = {}
        self.init_governance_environment()
        self.load_documents()
        self.load_reviews()
    
    def init_governance_environment(self):
        """Initialize governance environment."""
        print("📋 Initializing advanced governance environment...")
        
        # Create governance directories
        self.governance_dir = Path("governance")
        self.governance_dir.mkdir(exist_ok=True)
        
        self.policies_dir = Path("governance_policies")
        self.policies_dir.mkdir(exist_ok=True)
        
        self.reviews_dir = Path("governance_reviews")
        self.reviews_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.init_governance_database()
        
        print("✅ Advanced governance environment initialized")
    
    def init_governance_database(self):
        """Initialize governance database."""
        conn = sqlite3.connect("governance.db")
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS governance_documents (
                id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                governance_type TEXT,
                level TEXT,
                category TEXT,
                owner TEXT,
                approver TEXT,
                version TEXT,
                status TEXT,
                effective_date DATETIME,
                review_date DATETIME,
                content TEXT,
                tags TEXT,
                dependencies TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS governance_reviews (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                reviewer TEXT,
                review_date DATETIME,
                status TEXT,
                comments TEXT,
                recommendations TEXT,
                next_review DATETIME,
                approved BOOLEAN,
                FOREIGN KEY (document_id) REFERENCES governance_documents (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_documents(self):
        """Load existing governance documents."""
        conn = sqlite3.connect("governance.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM governance_documents")
        rows = cursor.fetchall()
        
        for row in rows:
            document = GovernanceDocument(
                id=row[0],
                title=row[1],
                description=row[2],
                governance_type=GovernanceType(row[3]),
                level=GovernanceLevel(row[4]),
                category=row[5],
                owner=row[6],
                approver=row[7],
                version=row[8],
                status=GovernanceStatus(row[9]),
                effective_date=datetime.fromisoformat(row[10]),
                review_date=datetime.fromisoformat(row[11]),
                content=row[12],
                tags=json.loads(row[13]) if row[13] else [],
                dependencies=json.loads(row[14]) if row[14] else [],
                created_at=datetime.fromisoformat(row[15])
            )
            self.documents[document.id] = document
        
        conn.close()
        
        # Create default documents if none exist
        if not self.documents:
            self.create_default_documents()
        
        print(f"✅ Loaded {len(self.documents)} governance documents")
    
    def load_reviews(self):
        """Load existing governance reviews."""
        conn = sqlite3.connect("governance.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM governance_reviews")
        rows = cursor.fetchall()
        
        for row in rows:
            review = GovernanceReview(
                id=row[0],
                document_id=row[1],
                reviewer=row[2],
                review_date=datetime.fromisoformat(row[3]),
                status=row[4],
                comments=row[5],
                recommendations=json.loads(row[6]) if row[6] else [],
                next_review=datetime.fromisoformat(row[7]),
                approved=bool(row[8])
            )
            self.reviews[review.id] = review
        
        conn.close()
        
        print(f"✅ Loaded {len(self.reviews)} governance reviews")
    
    def create_default_documents(self):
        """Create default governance documents."""
        default_documents = [
            {
                'id': 'data_governance_policy',
                'title': 'Data Governance Policy',
                'description': 'Comprehensive data governance policy for the organization',
                'governance_type': GovernanceType.POLICY,
                'level': GovernanceLevel.STRATEGIC,
                'category': 'Data Management',
                'owner': 'Chief Data Officer',
                'approver': 'CEO',
                'version': '1.0',
                'status': GovernanceStatus.ACTIVE,
                'effective_date': datetime.now(),
                'review_date': datetime.now() + timedelta(days=365),
                'content': 'This policy establishes the framework for data governance...',
                'tags': ['data', 'governance', 'policy'],
                'dependencies': []
            },
            {
                'id': 'ai_governance_framework',
                'title': 'AI Governance Framework',
                'description': 'Framework for governing artificial intelligence systems',
                'governance_type': GovernanceType.FRAMEWORK,
                'level': GovernanceLevel.STRATEGIC,
                'category': 'AI Management',
                'owner': 'Chief AI Officer',
                'approver': 'CTO',
                'version': '1.0',
                'status': GovernanceStatus.ACTIVE,
                'effective_date': datetime.now(),
                'review_date': datetime.now() + timedelta(days=180),
                'content': 'This framework provides guidelines for AI governance...',
                'tags': ['ai', 'governance', 'framework'],
                'dependencies': ['data_governance_policy']
            },
            {
                'id': 'security_governance_standard',
                'title': 'Security Governance Standard',
                'description': 'Standard for security governance and management',
                'governance_type': GovernanceType.STANDARD,
                'level': GovernanceLevel.TACTICAL,
                'category': 'Security',
                'owner': 'CISO',
                'approver': 'CTO',
                'version': '2.0',
                'status': GovernanceStatus.ACTIVE,
                'effective_date': datetime.now(),
                'review_date': datetime.now() + timedelta(days=90),
                'content': 'This standard defines security governance requirements...',
                'tags': ['security', 'governance', 'standard'],
                'dependencies': []
            }
        ]
        
        for doc_data in default_documents:
            self.create_document(
                document_id=doc_data['id'],
                title=doc_data['title'],
                description=doc_data['description'],
                governance_type=doc_data['governance_type'],
                level=doc_data['level'],
                category=doc_data['category'],
                owner=doc_data['owner'],
                approver=doc_data['approver'],
                version=doc_data['version'],
                status=doc_data['status'],
                effective_date=doc_data['effective_date'],
                review_date=doc_data['review_date'],
                content=doc_data['content'],
                tags=doc_data['tags'],
                dependencies=doc_data['dependencies']
            )
    
    def create_document(self, document_id: str, title: str, description: str,
                       governance_type: GovernanceType, level: GovernanceLevel,
                       category: str, owner: str, approver: str, version: str,
                       status: GovernanceStatus, effective_date: datetime,
                       review_date: datetime, content: str, tags: List[str],
                       dependencies: List[str]) -> GovernanceDocument:
        """Create a new governance document."""
        document = GovernanceDocument(
            id=document_id,
            title=title,
            description=description,
            governance_type=governance_type,
            level=level,
            category=category,
            owner=owner,
            approver=approver,
            version=version,
            status=status,
            effective_date=effective_date,
            review_date=review_date,
            content=content,
            tags=tags,
            dependencies=dependencies,
            created_at=datetime.now()
        )
        
        self.documents[document_id] = document
        
        # Save to database
        conn = sqlite3.connect("governance.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO governance_documents 
            (id, title, description, governance_type, level, category, owner, 
             approver, version, status, effective_date, review_date, content, 
             tags, dependencies, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (document_id, title, description, governance_type.value, level.value,
              category, owner, approver, version, status.value,
              effective_date.isoformat(), review_date.isoformat(), content,
              json.dumps(tags), json.dumps(dependencies), document.created_at.isoformat()))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Created governance document: {title}")
        return document
    
    def create_review(self, review_id: str, document_id: str, reviewer: str,
                     status: str, comments: str, recommendations: List[str]) -> GovernanceReview:
        """Create a new governance review."""
        if document_id not in self.documents:
            raise ValueError(f"Document {document_id} not found")
        
        document = self.documents[document_id]
        
        # Calculate next review date
        next_review = self._calculate_next_review(document.review_date)
        
        review = GovernanceReview(
            id=review_id,
            document_id=document_id,
            reviewer=reviewer,
            review_date=datetime.now(),
            status=status,
            comments=comments,
            recommendations=recommendations,
            next_review=next_review,
            approved=status.lower() == 'approved'
        )
        
        self.reviews[review_id] = review
        
        # Save to database
        conn = sqlite3.connect("governance.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO governance_reviews 
            (id, document_id, reviewer, review_date, status, comments, 
             recommendations, next_review, approved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (review_id, document_id, reviewer, review.review_date.isoformat(),
              status, comments, json.dumps(recommendations),
              next_review.isoformat(), review.approved))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Created governance review: {document.title}")
        return review
    
    def get_governance_stats(self) -> Dict[str, Any]:
        """Get governance statistics."""
        total_documents = len(self.documents)
        total_reviews = len(self.reviews)
        
        # Status counts
        status_counts = defaultdict(int)
        for doc in self.documents.values():
            status_counts[doc.status] += 1
        
        # Type counts
        type_counts = defaultdict(int)
        for doc in self.documents.values():
            type_counts[doc.governance_type] += 1
        
        # Level counts
        level_counts = defaultdict(int)
        for doc in self.documents.values():
            level_counts[doc.level] += 1
        
        # Overdue reviews
        overdue_count = 0
        for doc in self.documents.values():
            if doc.review_date < datetime.now():
                overdue_count += 1
        
        return {
            'total_documents': total_documents,
            'total_reviews': total_reviews,
            'status_counts': {k.value: v for k, v in status_counts.items()},
            'type_counts': {k.value: v for k, v in type_counts.items()},
            'level_counts': {k.value: v for k, v in level_counts.items()},
            'overdue_reviews': overdue_count
        }
    
    def generate_governance_report(self) -> str:
        """Generate governance report."""
        stats = self.get_governance_stats()
        
        report = f"""
BUL Advanced Governance Manager Report
=====================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DOCUMENTS
---------
Total Documents: {stats['total_documents']}
"""
        
        # Group by type
        by_type = defaultdict(list)
        for doc in self.documents.values():
            by_type[doc.governance_type].append(doc)
        
        for doc_type, docs in by_type.items():
            report += f"""
{doc_type.value.title()} Documents ({len(docs)}):
"""
            for doc in docs:
                report += f"""
{doc.title} ({doc.id}):
  Level: {doc.level.value}
  Category: {doc.category}
  Owner: {doc.owner}
  Status: {doc.status.value}
  Version: {doc.version}
  Effective Date: {doc.effective_date.strftime('%Y-%m-%d')}
  Review Date: {doc.review_date.strftime('%Y-%m-%d')}
  Tags: {', '.join(doc.tags)}
"""
        
        # Recent reviews
        recent_reviews = sorted(
            self.reviews.values(),
            key=lambda x: x.review_date,
            reverse=True
        )[:10]
        
        if recent_reviews:
            report += f"""
RECENT REVIEWS
--------------
"""
            for review in recent_reviews:
                doc = self.documents[review.document_id]
                report += f"""
{doc.title} - {review.review_date.strftime('%Y-%m-%d')}
  Reviewer: {review.reviewer}
  Status: {review.status}
  Approved: {review.approved}
  Comments: {review.comments[:100]}...
"""
        
        return report
    
    def _calculate_next_review(self, current_review_date: datetime) -> datetime:
        """Calculate next review date."""
        # Default to annual review
        return current_review_date + timedelta(days=365)

def main():
    """Main governance manager function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Advanced Governance Manager")
    parser.add_argument("--create-document", help="Create a new governance document")
    parser.add_argument("--create-review", help="Create a new governance review")
    parser.add_argument("--list-documents", action="store_true", help="List all documents")
    parser.add_argument("--list-reviews", action="store_true", help="List all reviews")
    parser.add_argument("--stats", action="store_true", help="Show governance statistics")
    parser.add_argument("--report", action="store_true", help="Generate governance report")
    parser.add_argument("--title", help="Title for document")
    parser.add_argument("--description", help="Description for document")
    parser.add_argument("--governance-type", choices=['policy', 'procedure', 'standard', 'guideline', 'framework'],
                       help="Governance type")
    parser.add_argument("--level", choices=['strategic', 'tactical', 'operational'],
                       help="Governance level")
    parser.add_argument("--category", help="Category for document")
    parser.add_argument("--owner", help="Document owner")
    parser.add_argument("--approver", help="Document approver")
    parser.add_argument("--version", help="Document version")
    parser.add_argument("--status", choices=['active', 'inactive', 'pending', 'archived'],
                       help="Document status")
    parser.add_argument("--content", help="Document content")
    parser.add_argument("--tags", help="Comma-separated list of tags")
    parser.add_argument("--document-id", help="Document ID for review")
    parser.add_argument("--reviewer", help="Reviewer name")
    parser.add_argument("--comments", help="Review comments")
    parser.add_argument("--recommendations", help="Comma-separated list of recommendations")
    
    args = parser.parse_args()
    
    governance_manager = AdvancedGovernanceManager()
    
    print("📋 BUL Advanced Governance Manager")
    print("=" * 40)
    
    if args.create_document:
        if not all([args.title, args.description, args.governance_type, args.level, args.category, args.owner, args.approver]):
            print("❌ Error: --title, --description, --governance-type, --level, --category, --owner, and --approver are required")
            return 1
        
        tags = args.tags.split(',') if args.tags else []
        
        document = governance_manager.create_document(
            document_id=args.create_document,
            title=args.title,
            description=args.description,
            governance_type=GovernanceType(args.governance_type),
            level=GovernanceLevel(args.level),
            category=args.category,
            owner=args.owner,
            approver=args.approver,
            version=args.version or "1.0",
            status=GovernanceStatus(args.status) if args.status else GovernanceStatus.ACTIVE,
            effective_date=datetime.now(),
            review_date=datetime.now() + timedelta(days=365),
            content=args.content or "",
            tags=tags,
            dependencies=[]
        )
        print(f"✅ Created document: {document.title}")
    
    elif args.create_review:
        if not all([args.document_id, args.reviewer, args.comments]):
            print("❌ Error: --document-id, --reviewer, and --comments are required")
            return 1
        
        recommendations = args.recommendations.split(',') if args.recommendations else []
        
        review = governance_manager.create_review(
            review_id=args.create_review,
            document_id=args.document_id,
            reviewer=args.reviewer,
            status="completed",
            comments=args.comments,
            recommendations=recommendations
        )
        print(f"✅ Created review: {review.id}")
    
    elif args.list_documents:
        documents = governance_manager.documents
        if documents:
            print(f"\n📋 Governance Documents ({len(documents)}):")
            print("-" * 60)
            for doc_id, doc in documents.items():
                print(f"{doc.title} ({doc_id}):")
                print(f"  Type: {doc.governance_type.value}")
                print(f"  Level: {doc.level.value}")
                print(f"  Owner: {doc.owner}")
                print(f"  Status: {doc.status.value}")
                print(f"  Version: {doc.version}")
                print()
        else:
            print("No documents found.")
    
    elif args.list_reviews:
        reviews = governance_manager.reviews
        if reviews:
            print(f"\n📋 Governance Reviews ({len(reviews)}):")
            print("-" * 60)
            for review_id, review in reviews.items():
                doc = governance_manager.documents[review.document_id]
                print(f"{doc.title} - {review.review_date.strftime('%Y-%m-%d')} ({review_id}):")
                print(f"  Reviewer: {review.reviewer}")
                print(f"  Status: {review.status}")
                print(f"  Approved: {review.approved}")
                print()
        else:
            print("No reviews found.")
    
    elif args.stats:
        stats = governance_manager.get_governance_stats()
        print(f"\n📊 Governance Statistics:")
        print(f"   Total Documents: {stats['total_documents']}")
        print(f"   Total Reviews: {stats['total_reviews']}")
        print(f"   Overdue Reviews: {stats['overdue_reviews']}")
        print(f"   Status Distribution:")
        for status, count in stats['status_counts'].items():
            print(f"     {status.title()}: {count}")
        print(f"   Type Distribution:")
        for doc_type, count in stats['type_counts'].items():
            print(f"     {doc_type.title()}: {count}")
        print(f"   Level Distribution:")
        for level, count in stats['level_counts'].items():
            print(f"     {level.title()}: {count}")
    
    elif args.report:
        report = governance_manager.generate_governance_report()
        print(report)
        
        # Save report
        report_file = f"governance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_file}")
    
    else:
        # Show quick overview
        stats = governance_manager.get_governance_stats()
        print(f"📋 Documents: {stats['total_documents']}")
        print(f"📋 Reviews: {stats['total_reviews']}")
        print(f"⚠️ Overdue Reviews: {stats['overdue_reviews']}")
        print(f"\n💡 Use --list-documents to see all documents")
        print(f"💡 Use --create-document to create a new document")
        print(f"💡 Use --report to generate governance report")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)