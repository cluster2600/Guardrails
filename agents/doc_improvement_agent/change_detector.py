"""
Change Detector - Detects code changes that impact documentation.

This module analyzes git diffs to identify code changes that might require
documentation updates, mapping them to specific documentation files.
"""

import re
import subprocess
from fnmatch import fnmatch
from pathlib import Path
from typing import List, Set, Dict, Optional, Tuple
from datetime import datetime

from .models import ChangeImpact, ChangeType, Severity
from .config import AgentConfig, CodeToDocsMapping


class ChangeDetector:
    """Detects code changes that impact documentation."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.repo_path = config.repo_path
    
    def detect_changes(self, base_ref: str = "main") -> List[ChangeImpact]:
        """
        Detect code changes between current HEAD and base_ref.
        
        Args:
            base_ref: Git reference to compare against (default: main)
            
        Returns:
            List of ChangeImpact objects describing documentation impact
        """
        # Get list of changed files
        changed_files = self._get_changed_files(base_ref)
        
        if not changed_files:
            return []
        
        impacts = []
        for file_path in changed_files:
            # Check if this file affects customer-facing surface
            if self._is_customer_facing(file_path):
                impact = self._analyze_file_impact(file_path, base_ref)
                if impact and impact.impacted_docs:
                    impacts.append(impact)
        
        return impacts
    
    def _get_changed_files(self, base_ref: str) -> List[str]:
        """Get list of files changed since base_ref."""
        try:
            # First, try to compare with remote tracking branch
            result = subprocess.run(
                ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
            )
            
            if result.returncode != 0:
                # Fall back to simple diff
                result = subprocess.run(
                    ["git", "diff", "--name-only", base_ref],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path,
                )
            
            if result.returncode == 0:
                return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
            
            return []
            
        except Exception as e:
            print(f"Warning: Could not get changed files: {e}")
            return []
    
    def _is_customer_facing(self, file_path: str) -> bool:
        """
        Determine if a file change affects customer-facing surface.
        
        Customer-facing includes:
        - API endpoints
        - Public interfaces
        - Examples
        - Configuration schemas
        - CLI commands
        """
        for pattern in self.config.customer_facing_patterns:
            if self._matches_pattern(file_path, pattern):
                return True
        return False
    
    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if a file path matches a glob pattern."""
        # Handle ** patterns
        if "**" in pattern:
            # Convert ** to regex-friendly pattern
            regex_pattern = pattern.replace("**", ".*").replace("*", "[^/]*")
            return bool(re.match(regex_pattern, file_path))
        else:
            return fnmatch(file_path, pattern)
    
    def _analyze_file_impact(self, file_path: str, base_ref: str) -> Optional[ChangeImpact]:
        """Analyze the impact of changes to a specific file."""
        
        # Determine change type
        change_type = self._classify_change_type(file_path)
        
        # Find impacted documentation
        impacted_docs = self._find_impacted_docs(file_path, change_type)
        
        # Get changed symbols (functions, classes, etc.)
        changed_symbols = self._extract_changed_symbols(file_path, base_ref)
        
        # Determine severity
        severity = self._determine_severity(file_path, change_type, changed_symbols)
        
        # Get diff summary
        diff_summary = self._get_diff_summary(file_path, base_ref)
        
        return ChangeImpact(
            source_file=file_path,
            impacted_docs=impacted_docs,
            change_type=change_type,
            severity=severity,
            changed_symbols=changed_symbols,
            git_diff_summary=diff_summary,
            detected_at=datetime.now(),
        )
    
    def _classify_change_type(self, file_path: str) -> ChangeType:
        """Classify the type of change based on file path."""
        path_lower = file_path.lower()
        
        if "api" in path_lower or "endpoint" in path_lower:
            return ChangeType.API
        elif "config" in path_lower:
            return ChangeType.CONFIG
        elif "schema" in path_lower:
            return ChangeType.SCHEMA
        elif "example" in path_lower or "sample" in path_lower:
            return ChangeType.EXAMPLE
        elif "cli" in path_lower or "command" in path_lower:
            return ChangeType.CLI
        elif "model" in path_lower:
            return ChangeType.MODEL
        elif "util" in path_lower or "helper" in path_lower:
            return ChangeType.UTILITY
        else:
            return ChangeType.UNKNOWN
    
    def _find_impacted_docs(self, file_path: str, change_type: ChangeType) -> List[str]:
        """Find documentation files that might be impacted by this change."""
        impacted = set()
        
        # Check explicit mappings first
        for mapping in self.config.code_to_docs_mappings:
            if self._matches_pattern(file_path, mapping.pattern):
                for doc_pattern in mapping.impacts:
                    # Find actual doc files matching the pattern
                    matching_docs = self._find_docs_matching_pattern(doc_pattern)
                    impacted.update(matching_docs)
        
        # Also check for docs that reference this file
        referencing_docs = self._find_docs_referencing_file(file_path)
        impacted.update(referencing_docs)
        
        return list(impacted)
    
    def _find_docs_matching_pattern(self, pattern: str) -> List[str]:
        """Find documentation files matching a glob pattern."""
        docs = []
        
        for docs_dir in self.config.docs_directories:
            docs_path = self.repo_path / docs_dir
            if not docs_path.exists():
                continue
            
            for doc_file in docs_path.rglob("*.md"):
                relative_path = str(doc_file.relative_to(self.repo_path))
                if self._matches_pattern(relative_path, pattern):
                    docs.append(relative_path)
            
            for doc_file in docs_path.rglob("*.rst"):
                relative_path = str(doc_file.relative_to(self.repo_path))
                if self._matches_pattern(relative_path, pattern):
                    docs.append(relative_path)
        
        return docs
    
    def _find_docs_referencing_file(self, file_path: str) -> List[str]:
        """Find docs that reference the given source file."""
        referencing = []
        file_name = Path(file_path).name
        file_stem = Path(file_path).stem
        
        for docs_dir in self.config.docs_directories:
            docs_path = self.repo_path / docs_dir
            if not docs_path.exists():
                continue
            
            for doc_file in docs_path.rglob("*.md"):
                try:
                    content = doc_file.read_text(encoding='utf-8')
                    # Check if doc references this file
                    if file_name in content or file_stem in content or file_path in content:
                        referencing.append(str(doc_file.relative_to(self.repo_path)))
                except Exception:
                    continue
        
        return referencing
    
    def _extract_changed_symbols(self, file_path: str, base_ref: str) -> List[str]:
        """Extract names of changed functions, classes, etc."""
        symbols = []
        
        try:
            # Get the diff for this file
            result = subprocess.run(
                ["git", "diff", "-U0", base_ref, "--", file_path],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
            )
            
            if result.returncode != 0:
                return symbols
            
            diff_content = result.stdout
            
            # Extract Python function/class definitions from diff
            # Patterns for added/modified lines
            python_patterns = [
                r'^\+\s*def\s+(\w+)',           # Function definitions
                r'^\+\s*class\s+(\w+)',          # Class definitions
                r'^\+\s*async\s+def\s+(\w+)',   # Async function definitions
            ]
            
            for pattern in python_patterns:
                matches = re.findall(pattern, diff_content, re.MULTILINE)
                symbols.extend(matches)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_symbols = []
            for s in symbols:
                if s not in seen:
                    seen.add(s)
                    unique_symbols.append(s)
            
            return unique_symbols
            
        except Exception as e:
            print(f"Warning: Could not extract symbols from {file_path}: {e}")
            return symbols
    
    def _determine_severity(
        self, 
        file_path: str, 
        change_type: ChangeType,
        changed_symbols: List[str]
    ) -> Severity:
        """Determine the severity of documentation impact."""
        
        # Check explicit mapping severity
        for mapping in self.config.code_to_docs_mappings:
            if self._matches_pattern(file_path, mapping.pattern):
                severity_str = mapping.severity.lower()
                if severity_str == "critical":
                    return Severity.CRITICAL
                elif severity_str == "high":
                    return Severity.HIGH
                elif severity_str == "medium":
                    return Severity.MEDIUM
                elif severity_str == "low":
                    return Severity.LOW
        
        # Infer from change type
        if change_type in [ChangeType.API, ChangeType.CLI, ChangeType.SCHEMA]:
            return Severity.HIGH
        elif change_type in [ChangeType.CONFIG, ChangeType.EXAMPLE]:
            return Severity.MEDIUM
        else:
            return Severity.LOW
    
    def _get_diff_summary(self, file_path: str, base_ref: str) -> str:
        """Get a summary of the diff for a file."""
        try:
            result = subprocess.run(
                ["git", "diff", "--stat", base_ref, "--", file_path],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            return ""
            
        except Exception:
            return ""
    
    def get_all_docs(self) -> List[str]:
        """Get all documentation files in the repository."""
        all_docs = []
        
        for docs_dir in self.config.docs_directories:
            docs_path = self.repo_path / docs_dir
            if not docs_path.exists():
                continue
            
            for doc_file in docs_path.rglob("*.md"):
                all_docs.append(str(doc_file.relative_to(self.repo_path)))
            
            for doc_file in docs_path.rglob("*.rst"):
                all_docs.append(str(doc_file.relative_to(self.repo_path)))
        
        return all_docs
    
    def full_scan(self) -> List[ChangeImpact]:
        """
        Perform a full scan of all documentation without comparing to base ref.
        
        This is useful for scheduled runs that want to audit all docs.
        """
        all_docs = self.get_all_docs()
        
        # Create synthetic impacts for each doc
        impacts = []
        for doc_path in all_docs:
            impact = ChangeImpact(
                source_file="(full scan)",
                impacted_docs=[doc_path],
                change_type=ChangeType.UNKNOWN,
                severity=Severity.MEDIUM,
                changed_symbols=[],
                git_diff_summary="Full documentation scan",
                detected_at=datetime.now(),
            )
            impacts.append(impact)
        
        return impacts
