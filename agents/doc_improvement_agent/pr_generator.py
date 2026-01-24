"""
PR Generator - Creates pull requests with documentation improvements.

This module handles Git operations and PR creation for both GitHub and GitLab.
"""

import os
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .models import ImprovementReport, DocImprovement, RuleResult


@dataclass
class PRInfo:
    """Information about a created pull request."""
    url: str
    number: int
    branch: str
    title: str
    state: str = "open"


class PRGenerator:
    """Generates pull requests with documentation improvements."""
    
    def __init__(
        self, 
        repo_path: Path,
        github_token: Optional[str] = None,
        gitlab_token: Optional[str] = None,
        branch_prefix: str = "docs/auto-improvement",
    ):
        self.repo_path = repo_path
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN')
        self.gitlab_token = gitlab_token or os.environ.get('GITLAB_TOKEN')
        self.branch_prefix = branch_prefix
    
    def create_branch(self, branch_name: Optional[str] = None) -> str:
        """Create a new branch for the improvements."""
        if not branch_name:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            branch_name = f"{self.branch_prefix}-{timestamp}"
        
        # Ensure we're on latest main first
        self._run_git("fetch", "origin")
        
        # Create and checkout new branch
        self._run_git("checkout", "-b", branch_name)
        
        return branch_name
    
    def commit_changes(
        self, 
        message: str,
        files: Optional[List[str]] = None,
    ) -> bool:
        """Commit changes to the current branch."""
        try:
            if files:
                for f in files:
                    self._run_git("add", f)
            else:
                self._run_git("add", ".")
            
            # Check if there are changes to commit
            result = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=self.repo_path,
            )
            
            if result.returncode == 0:
                # No changes to commit
                return False
            
            self._run_git("commit", "-m", message)
            return True
            
        except Exception as e:
            print(f"Warning: Could not commit changes: {e}")
            return False
    
    def push_branch(self, branch_name: str) -> bool:
        """Push the branch to remote."""
        try:
            self._run_git("push", "-u", "origin", branch_name)
            return True
        except Exception as e:
            print(f"Warning: Could not push branch: {e}")
            return False
    
    def create_pr(
        self,
        branch_name: str,
        title: str,
        body: str,
        base_branch: str = "main",
        draft: bool = False,
    ) -> Optional[PRInfo]:
        """Create a pull request using gh CLI or GitLab API."""
        
        # Try GitHub first (gh CLI)
        pr_info = self._create_github_pr(branch_name, title, body, base_branch, draft)
        if pr_info:
            return pr_info
        
        # Fall back to GitLab
        pr_info = self._create_gitlab_mr(branch_name, title, body, base_branch, draft)
        if pr_info:
            return pr_info
        
        print("Warning: Could not create PR. Please create manually.")
        return None
    
    def _create_github_pr(
        self,
        branch_name: str,
        title: str,
        body: str,
        base_branch: str,
        draft: bool,
    ) -> Optional[PRInfo]:
        """Create a GitHub PR using gh CLI."""
        try:
            cmd = [
                "gh", "pr", "create",
                "--title", title,
                "--body", body,
                "--base", base_branch,
                "--head", branch_name,
            ]
            
            if draft:
                cmd.append("--draft")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.repo_path,
            )
            
            if result.returncode == 0:
                # Extract PR URL from output
                url = result.stdout.strip()
                
                # Get PR number from URL
                try:
                    number = int(url.split('/')[-1])
                except (ValueError, IndexError):
                    number = 0
                
                return PRInfo(
                    url=url,
                    number=number,
                    branch=branch_name,
                    title=title,
                )
            
            return None
            
        except FileNotFoundError:
            # gh CLI not installed
            return None
        except Exception as e:
            print(f"Warning: GitHub PR creation failed: {e}")
            return None
    
    def _create_gitlab_mr(
        self,
        branch_name: str,
        title: str,
        body: str,
        base_branch: str,
        draft: bool,
    ) -> Optional[PRInfo]:
        """Create a GitLab merge request using glab CLI or API."""
        try:
            # Try glab CLI first
            cmd = [
                "glab", "mr", "create",
                "--title", title if not draft else f"Draft: {title}",
                "--description", body,
                "--target-branch", base_branch,
                "--source-branch", branch_name,
                "--no-editor",
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.repo_path,
            )
            
            if result.returncode == 0:
                # Extract MR URL from output
                output_lines = result.stdout.strip().split('\n')
                url = ""
                for line in output_lines:
                    if "http" in line:
                        url = line.strip()
                        break
                
                # Get MR number from URL
                try:
                    number = int(url.split('/')[-1])
                except (ValueError, IndexError):
                    number = 0
                
                return PRInfo(
                    url=url,
                    number=number,
                    branch=branch_name,
                    title=title,
                )
            
            return None
            
        except FileNotFoundError:
            # glab CLI not installed
            return None
        except Exception as e:
            print(f"Warning: GitLab MR creation failed: {e}")
            return None
    
    def _run_git(self, *args) -> subprocess.CompletedProcess:
        """Run a git command and return the result."""
        result = subprocess.run(
            ["git"] + list(args),
            capture_output=True,
            text=True,
            cwd=self.repo_path,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Git command failed: {result.stderr}")
        
        return result
    
    def generate_pr_body(
        self, 
        report: ImprovementReport,
        improvements: List[DocImprovement],
    ) -> str:
        """Generate a comprehensive PR body from the improvement report."""
        
        lines = [
            "## Summary",
            "",
            f"This PR contains automated documentation improvements based on code changes.",
            "",
            f"- **Trigger**: {report.trigger}",
            f"- **Base Ref**: {report.base_ref}",
            f"- **Run ID**: {report.run_id}",
            "",
        ]
        
        # Changes detected section
        if report.changes_detected:
            lines.extend([
                "## Code Changes Detected",
                "",
            ])
            for change in report.changes_detected[:5]:  # Limit to 5
                lines.append(f"- `{change.source_file}` ({change.change_type.value}, {change.severity.value})")
                if change.changed_symbols:
                    lines.append(f"  - Symbols: {', '.join(change.changed_symbols[:3])}")
            
            if len(report.changes_detected) > 5:
                lines.append(f"- ... and {len(report.changes_detected) - 5} more")
            
            lines.append("")
        
        # Impacted docs section
        if report.docs_impacted:
            lines.extend([
                "## Documentation Updated",
                "",
            ])
            for doc in report.docs_impacted[:10]:
                lines.append(f"- `{doc}`")
            
            if len(report.docs_impacted) > 10:
                lines.append(f"- ... and {len(report.docs_impacted) - 10} more")
            
            lines.append("")
        
        # Rules applied section
        if report.rules_executed:
            lines.extend([
                "## Rules Applied",
                "",
            ])
            
            rule_summary: Dict[str, int] = {}
            for result in report.rules_executed:
                rule_summary[result.rule_name] = rule_summary.get(result.rule_name, 0) + 1
            
            for rule_name, count in rule_summary.items():
                lines.append(f"- **{rule_name}**: {count} files")
            
            lines.append("")
        
        # Results section
        lines.extend([
            "## Results",
            "",
            f"- **Total Issues Found**: {report.total_issues_found}",
            f"- **Issues Fixed**: {report.total_issues_fixed}",
            f"- **Average Confidence**: {report.average_confidence:.1f}%",
            "",
        ])
        
        # Confidence indicator
        confidence = report.average_confidence
        if confidence >= 90:
            lines.append("🟢 **Confidence Level**: High")
        elif confidence >= 70:
            lines.append("🟡 **Confidence Level**: Moderate")
        elif confidence >= 50:
            lines.append("🟠 **Confidence Level**: Low - Manual review recommended")
        else:
            lines.append("🔴 **Confidence Level**: Uncertain - Careful review required")
        
        lines.append("")
        
        # Review checklist
        lines.extend([
            "## Review Checklist",
            "",
            "- [ ] Technical accuracy verified",
            "- [ ] Code examples still work",
            "- [ ] No sensitive information exposed",
            "- [ ] Links are valid",
            "- [ ] Style guide compliance checked",
            "",
        ])
        
        # Footer
        lines.extend([
            "---",
            "",
            "*This PR was generated by the Documentation Self-Improvement Agent.*",
            "",
        ])
        
        return "\n".join(lines)
    
    def generate_commit_message(
        self,
        improvements: List[DocImprovement],
        rules_applied: List[str],
    ) -> str:
        """Generate a descriptive commit message."""
        
        # Count affected files
        files_changed = len(improvements)
        
        # Summarize rules
        if "validation/docs-audit-enhanced" in rules_applied:
            main_action = "audit and improve"
        elif "transformation/docs-polish" in rules_applied:
            main_action = "polish"
        else:
            main_action = "update"
        
        # Short summary
        if files_changed == 1:
            title = f"docs: {main_action} {improvements[0].doc_path}"
        else:
            title = f"docs: {main_action} {files_changed} documentation files"
        
        # Body
        body_lines = [
            "",
            "Automated documentation improvements:",
            "",
        ]
        
        for rule in rules_applied:
            body_lines.append(f"- Applied {rule}")
        
        body_lines.extend([
            "",
            f"Files updated: {files_changed}",
        ])
        
        return title + "\n" + "\n".join(body_lines)
    
    def cleanup_branch(self, branch_name: str, return_to: str = "main") -> None:
        """Clean up by returning to the original branch."""
        try:
            self._run_git("checkout", return_to)
        except Exception:
            pass
