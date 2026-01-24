"""
Documentation Self-Improvement Agent

An intelligent agent that detects code changes affecting documentation,
runs appropriate validation and improvement rules, and creates PRs with fixes.

Features:
- Automatic detection of code changes that impact docs
- Intelligent routing to appropriate rules (audit-deep, polish, etc.)
- Multi-trigger support: CI/CD, scheduled, user command, LLM-driven
- Automatic PR creation with detailed summaries

Usage:
    # CLI
    python -m agents.doc_improvement_agent --base-ref main --create-pr
    
    # As module
    from agents.doc_improvement_agent import DocImprovementAgent
    agent = DocImprovementAgent(repo_path=".")
    agent.run()
"""

from .agent import DocImprovementAgent
from .change_detector import ChangeDetector
from .rule_router import RuleRouter
from .pr_generator import PRGenerator
from .models import (
    ChangeImpact,
    DocImprovement,
    RuleResult,
    ImprovementReport,
)

__version__ = "0.1.0"
__all__ = [
    "DocImprovementAgent",
    "ChangeDetector", 
    "RuleRouter",
    "PRGenerator",
    "ChangeImpact",
    "DocImprovement",
    "RuleResult",
    "ImprovementReport",
]
