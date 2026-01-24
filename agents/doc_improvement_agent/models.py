"""
Data models for the Documentation Self-Improvement Agent.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional


class ChangeType(Enum):
    """Type of code change detected."""
    API = "api"
    CONFIG = "config"
    SCHEMA = "schema"
    EXAMPLE = "example"
    CLI = "cli"
    MODEL = "model"
    UTILITY = "utility"
    UNKNOWN = "unknown"


class Severity(Enum):
    """Severity of the documentation impact."""
    CRITICAL = "critical"  # Breaking changes, security
    HIGH = "high"          # API changes, new features
    MEDIUM = "medium"      # Config changes, examples
    LOW = "low"            # Minor updates, comments


class RuleCategory(Enum):
    """Category of documentation rules."""
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    CONTENT = "content"
    WORKFLOW = "workflow"


class ConfidenceLevel(Enum):
    """Confidence level for audit results."""
    HIGH = "high"        # 90-100%
    MODERATE = "moderate"  # 70-89%
    LOW = "low"          # 50-69%
    UNCERTAIN = "uncertain"  # 0-49%


@dataclass
class ChangeImpact:
    """Represents the impact of a code change on documentation."""
    source_file: str
    impacted_docs: List[str]
    change_type: ChangeType
    severity: Severity
    changed_symbols: List[str] = field(default_factory=list)
    git_diff_summary: str = ""
    detected_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "impacted_docs": self.impacted_docs,
            "change_type": self.change_type.value,
            "severity": self.severity.value,
            "changed_symbols": self.changed_symbols,
            "git_diff_summary": self.git_diff_summary,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class RuleResult:
    """Result from running a documentation rule."""
    rule_name: str
    rule_category: RuleCategory
    doc_path: str
    success: bool
    confidence_score: Optional[float] = None
    confidence_level: Optional[ConfidenceLevel] = None
    issues_found: List[Dict[str, Any]] = field(default_factory=list)
    improvements_made: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    raw_output: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "rule_category": self.rule_category.value,
            "doc_path": self.doc_path,
            "success": self.success,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level.value if self.confidence_level else None,
            "issues_found": self.issues_found,
            "improvements_made": self.improvements_made,
            "execution_time_seconds": self.execution_time_seconds,
        }


@dataclass
class DocImprovement:
    """A single improvement made to a documentation file."""
    doc_path: str
    original_content: str
    improved_content: str
    change_summary: str
    rules_applied: List[str]
    confidence_score: float
    issues_fixed: List[str] = field(default_factory=list)
    
    @property
    def has_changes(self) -> bool:
        return self.original_content != self.improved_content
    
    def get_diff_stats(self) -> Dict[str, int]:
        """Get basic diff statistics."""
        original_lines = self.original_content.splitlines()
        improved_lines = self.improved_content.splitlines()
        return {
            "lines_added": max(0, len(improved_lines) - len(original_lines)),
            "lines_removed": max(0, len(original_lines) - len(improved_lines)),
            "original_lines": len(original_lines),
            "improved_lines": len(improved_lines),
        }


@dataclass
class ImprovementReport:
    """Complete report of a documentation improvement run."""
    run_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    base_ref: str = "main"
    trigger: str = "manual"  # ci, scheduled, manual, llm
    
    # Detection results
    changes_detected: List[ChangeImpact] = field(default_factory=list)
    docs_impacted: List[str] = field(default_factory=list)
    
    # Execution results
    rules_executed: List[RuleResult] = field(default_factory=list)
    improvements: List[DocImprovement] = field(default_factory=list)
    
    # PR information
    pr_created: bool = False
    pr_url: Optional[str] = None
    branch_name: Optional[str] = None
    
    # Summary
    total_issues_found: int = 0
    total_issues_fixed: int = 0
    average_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "base_ref": self.base_ref,
            "trigger": self.trigger,
            "changes_detected": [c.to_dict() for c in self.changes_detected],
            "docs_impacted": self.docs_impacted,
            "rules_executed": [r.to_dict() for r in self.rules_executed],
            "pr_created": self.pr_created,
            "pr_url": self.pr_url,
            "branch_name": self.branch_name,
            "total_issues_found": self.total_issues_found,
            "total_issues_fixed": self.total_issues_fixed,
            "average_confidence": self.average_confidence,
        }
    
    def generate_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"# Documentation Improvement Report",
            f"",
            f"**Run ID**: {self.run_id}",
            f"**Trigger**: {self.trigger}",
            f"**Base Ref**: {self.base_ref}",
            f"",
            f"## Summary",
            f"",
            f"- **Changes Detected**: {len(self.changes_detected)}",
            f"- **Docs Impacted**: {len(self.docs_impacted)}",
            f"- **Rules Executed**: {len(self.rules_executed)}",
            f"- **Issues Found**: {self.total_issues_found}",
            f"- **Issues Fixed**: {self.total_issues_fixed}",
            f"- **Average Confidence**: {self.average_confidence:.1f}%",
            f"",
        ]
        
        if self.pr_created and self.pr_url:
            lines.extend([
                f"## Pull Request",
                f"",
                f"**URL**: {self.pr_url}",
                f"**Branch**: {self.branch_name}",
                f"",
            ])
        
        if self.docs_impacted:
            lines.extend([
                f"## Impacted Documentation",
                f"",
            ])
            for doc in self.docs_impacted:
                lines.append(f"- `{doc}`")
            lines.append("")
        
        return "\n".join(lines)
