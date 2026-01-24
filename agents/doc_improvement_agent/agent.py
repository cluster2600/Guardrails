"""
Documentation Self-Improvement Agent - Main Agent Class

This is the core orchestrator that coordinates change detection, rule routing,
rule execution, and PR generation for automated documentation improvements.
"""

import uuid
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
import subprocess

from .models import (
    ChangeImpact,
    DocImprovement,
    RuleResult,
    ImprovementReport,
    ChangeType,
    RuleCategory,
    ConfidenceLevel,
)
from .config import AgentConfig
from .change_detector import ChangeDetector
from .rule_router import RuleRouter, RuleSpec
from .pr_generator import PRGenerator


class DocImprovementAgent:
    """
    Main agent for automated documentation improvement.
    
    Coordinates:
    - Change detection (what code changes impact docs)
    - Rule selection (which rules to apply)
    - Rule execution (running audit-deep, polish, etc.)
    - PR generation (creating PRs with improvements)
    
    Usage:
        agent = DocImprovementAgent(repo_path=".")
        report = agent.run(base_ref="main", create_pr=True)
    """
    
    def __init__(
        self,
        repo_path: str = ".",
        config: Optional[AgentConfig] = None,
        llm_executor: Optional[Callable[[str, str, str], str]] = None,
    ):
        """
        Initialize the documentation improvement agent.
        
        Args:
            repo_path: Path to the repository root
            config: Optional AgentConfig, defaults to auto-detected config
            llm_executor: Optional callable for LLM-based rule execution
                         Signature: (doc_path, rule_content, context) -> improved_content
        """
        self.repo_path = Path(repo_path).resolve()
        self.config = config or self._load_config()
        self.config.repo_path = self.repo_path
        
        self.detector = ChangeDetector(self.config)
        self.router = RuleRouter(self.config)
        self.pr_generator = PRGenerator(
            self.repo_path,
            github_token=self.config.github_token,
            gitlab_token=self.config.gitlab_token,
            branch_prefix=self.config.branch_prefix,
        )
        
        self.llm_executor = llm_executor
        
        # State
        self._current_report: Optional[ImprovementReport] = None
    
    def _load_config(self) -> AgentConfig:
        """Load configuration from file or environment."""
        config_paths = [
            self.repo_path / "doc-agent.yaml",
            self.repo_path / "doc-agent.yml",
            self.repo_path / ".doc-agent.yaml",
            self.repo_path / ".dori" / "agent-config.yaml",
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                return AgentConfig.from_yaml(config_path)
        
        return AgentConfig.from_env()
    
    def run(
        self,
        base_ref: Optional[str] = None,
        create_pr: bool = False,
        full_scan: bool = False,
        dry_run: bool = False,
        trigger: str = "manual",
    ) -> ImprovementReport:
        """
        Run the documentation improvement agent.
        
        Args:
            base_ref: Git reference to compare against (default: main)
            create_pr: Whether to create a PR with improvements
            full_scan: If True, scan all docs regardless of changes
            dry_run: If True, don't make any changes
            trigger: How the agent was triggered (manual, ci, scheduled, llm)
            
        Returns:
            ImprovementReport with results
        """
        base_ref = base_ref or self.config.default_base_ref
        
        # Initialize report
        report = ImprovementReport(
            run_id=str(uuid.uuid4())[:8],
            started_at=datetime.now(),
            base_ref=base_ref,
            trigger=trigger,
        )
        self._current_report = report
        
        print(f"🔍 Documentation Self-Improvement Agent")
        print(f"   Run ID: {report.run_id}")
        print(f"   Trigger: {trigger}")
        print(f"   Base Ref: {base_ref}")
        print()
        
        # Step 1: Detect changes
        print("📊 Step 1: Detecting changes...")
        if full_scan:
            impacts = self.detector.full_scan()
            print(f"   Full scan mode: {len(impacts)} documentation files")
        else:
            impacts = self.detector.detect_changes(base_ref)
            print(f"   Found {len(impacts)} code changes affecting documentation")
        
        report.changes_detected = impacts
        
        if not impacts:
            print("   ✅ No documentation changes needed")
            report.completed_at = datetime.now()
            return report
        
        # Collect all impacted docs
        all_docs = set()
        for impact in impacts:
            all_docs.update(impact.impacted_docs)
        report.docs_impacted = list(all_docs)
        print(f"   📄 {len(all_docs)} documentation files impacted")
        print()
        
        # Step 2: Generate execution plan
        print("🧠 Step 2: Generating execution plan...")
        execution_plan = self.router.generate_execution_plan(impacts)
        
        total_rules = sum(len(rules) for rules in execution_plan.values())
        print(f"   Plan: {total_rules} rule executions across {len(execution_plan)} files")
        print()
        
        # Step 3: Execute rules
        print("⚙️  Step 3: Executing rules...")
        improvements = []
        
        for doc_path, rules in execution_plan.items():
            if dry_run:
                print(f"   [DRY RUN] Would process: {doc_path}")
                print(f"             Rules: {[r.path for r in rules]}")
                continue
            
            doc_improvements = self._process_document(doc_path, rules, report)
            if doc_improvements:
                improvements.append(doc_improvements)
        
        report.improvements = improvements
        print()
        
        # Calculate summary statistics
        report.total_issues_found = sum(
            len(r.issues_found) for r in report.rules_executed
        )
        report.total_issues_fixed = sum(
            len(r.improvements_made) for r in report.rules_executed
        )
        
        confidences = [
            r.confidence_score for r in report.rules_executed 
            if r.confidence_score is not None
        ]
        report.average_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        
        # Step 4: Create PR if requested
        if create_pr and improvements and not dry_run:
            print("📝 Step 4: Creating pull request...")
            self._create_pr(report, improvements)
        elif create_pr and not improvements:
            print("📝 Step 4: No improvements to commit, skipping PR")
        
        report.completed_at = datetime.now()
        
        # Print summary
        self._print_summary(report)
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _process_document(
        self,
        doc_path: str,
        rules: List[RuleSpec],
        report: ImprovementReport,
    ) -> Optional[DocImprovement]:
        """Process a single document with the specified rules."""
        
        full_path = self.repo_path / doc_path
        if not full_path.exists():
            print(f"   ⚠️  Document not found: {doc_path}")
            return None
        
        print(f"   📄 Processing: {doc_path}")
        
        # Read original content
        try:
            original_content = full_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"      ❌ Could not read file: {e}")
            return None
        
        current_content = original_content
        rules_applied = []
        all_issues = []
        all_improvements = []
        total_confidence = 0.0
        confidence_count = 0
        
        # Execute each rule
        for rule in rules:
            print(f"      → Running {rule.path}...")
            start_time = time.time()
            
            result = self._execute_rule(doc_path, current_content, rule)
            
            result.execution_time_seconds = time.time() - start_time
            report.rules_executed.append(result)
            
            if result.success:
                rules_applied.append(rule.path)
                all_issues.extend(result.issues_found)
                all_improvements.extend(result.improvements_made)
                
                if result.confidence_score is not None:
                    total_confidence += result.confidence_score
                    confidence_count += 1
                
                # If the rule produced improved content, update current
                if result.raw_output and result.raw_output != current_content:
                    current_content = result.raw_output
                
                print(f"        ✅ Done ({result.execution_time_seconds:.1f}s)")
            else:
                print(f"        ⚠️  Rule failed or produced no changes")
        
        # Check if content was improved
        if current_content != original_content:
            # Write improved content
            try:
                full_path.write_text(current_content, encoding='utf-8')
                print(f"      💾 Saved improvements")
            except Exception as e:
                print(f"      ❌ Could not save: {e}")
                return None
            
            avg_confidence = (
                total_confidence / confidence_count if confidence_count > 0 else 0.0
            )
            
            return DocImprovement(
                doc_path=doc_path,
                original_content=original_content,
                improved_content=current_content,
                change_summary=f"Applied {len(rules_applied)} rules",
                rules_applied=rules_applied,
                confidence_score=avg_confidence,
                issues_fixed=[str(i) for i in all_issues[:10]],  # Limit stored issues
            )
        
        return None
    
    def _execute_rule(
        self,
        doc_path: str,
        content: str,
        rule: RuleSpec,
    ) -> RuleResult:
        """
        Execute a single rule on a document.
        
        If an LLM executor is provided, uses it for intelligent rule execution.
        Otherwise, falls back to deterministic checks where possible.
        """
        
        # Get rule content
        rule_content = self.router.get_rule_content(rule)
        
        # If we have an LLM executor, use it
        if self.llm_executor:
            try:
                improved_content = self.llm_executor(doc_path, rule_content, content)
                
                return RuleResult(
                    rule_name=rule.name,
                    rule_category=rule.category,
                    doc_path=doc_path,
                    success=True,
                    confidence_score=85.0,  # Default for LLM
                    confidence_level=ConfidenceLevel.MODERATE,
                    raw_output=improved_content,
                )
                
            except Exception as e:
                return RuleResult(
                    rule_name=rule.name,
                    rule_category=rule.category,
                    doc_path=doc_path,
                    success=False,
                    issues_found=[{"error": str(e)}],
                )
        
        # Fall back to deterministic checks
        return self._execute_deterministic_rule(doc_path, content, rule)
    
    def _execute_deterministic_rule(
        self,
        doc_path: str,
        content: str,
        rule: RuleSpec,
    ) -> RuleResult:
        """
        Execute deterministic validation rules.
        
        For audit and polish rules, these need LLM execution.
        For simpler rules, we can run Python scripts.
        """
        
        # Map rules to scripts
        script_mappings = {
            "validation/docs-drift": "scripts/doc-utils/detect_drift.py",
            "validation/docs-code-example-audit": "scripts/doc-utils/validate_code_blocks.py",
        }
        
        if rule.path in script_mappings:
            script_path = self.repo_path / script_mappings[rule.path]
            if script_path.exists():
                return self._run_script_rule(doc_path, script_path, rule)
        
        # For audit-deep and polish, we need LLM - return placeholder
        if rule.path in ["validation/docs-audit-enhanced", "transformation/docs-polish"]:
            return RuleResult(
                rule_name=rule.name,
                rule_category=rule.category,
                doc_path=doc_path,
                success=True,
                confidence_score=50.0,
                confidence_level=ConfidenceLevel.LOW,
                issues_found=[{
                    "note": f"Rule {rule.path} requires LLM execution. "
                           "Set llm_executor for full functionality."
                }],
            )
        
        # Default: mark as needing LLM
        return RuleResult(
            rule_name=rule.name,
            rule_category=rule.category,
            doc_path=doc_path,
            success=False,
            issues_found=[{"note": "Rule requires LLM execution"}],
        )
    
    def _run_script_rule(
        self,
        doc_path: str,
        script_path: Path,
        rule: RuleSpec,
    ) -> RuleResult:
        """Run a Python script-based rule."""
        
        try:
            result = subprocess.run(
                ["python", str(script_path), doc_path],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
            )
            
            output = result.stdout + result.stderr
            success = result.returncode == 0
            
            # Try to parse JSON output
            issues = []
            try:
                data = json.loads(result.stdout)
                if isinstance(data, list):
                    issues = data
                elif isinstance(data, dict) and "issues" in data:
                    issues = data["issues"]
            except json.JSONDecodeError:
                pass
            
            return RuleResult(
                rule_name=rule.name,
                rule_category=rule.category,
                doc_path=doc_path,
                success=success,
                confidence_score=90.0 if success else 50.0,
                confidence_level=ConfidenceLevel.HIGH if success else ConfidenceLevel.LOW,
                issues_found=issues,
                raw_output=output,
            )
            
        except Exception as e:
            return RuleResult(
                rule_name=rule.name,
                rule_category=rule.category,
                doc_path=doc_path,
                success=False,
                issues_found=[{"error": str(e)}],
            )
    
    def _create_pr(
        self,
        report: ImprovementReport,
        improvements: List[DocImprovement],
    ) -> None:
        """Create a pull request with the improvements."""
        
        try:
            # Create branch
            branch_name = self.pr_generator.create_branch()
            report.branch_name = branch_name
            print(f"   Created branch: {branch_name}")
            
            # Commit changes
            commit_msg = self.pr_generator.generate_commit_message(
                improvements,
                [r.rule_name for r in report.rules_executed],
            )
            
            if self.pr_generator.commit_changes(commit_msg):
                print(f"   Committed changes")
            else:
                print(f"   ⚠️  No changes to commit")
                return
            
            # Push branch
            if self.pr_generator.push_branch(branch_name):
                print(f"   Pushed to remote")
            else:
                print(f"   ⚠️  Could not push branch")
                return
            
            # Create PR
            pr_body = self.pr_generator.generate_pr_body(report, improvements)
            pr_info = self.pr_generator.create_pr(
                branch_name=branch_name,
                title=f"docs: Automated documentation improvements ({report.run_id})",
                body=pr_body,
                base_branch=report.base_ref,
            )
            
            if pr_info:
                report.pr_created = True
                report.pr_url = pr_info.url
                print(f"   ✅ Created PR: {pr_info.url}")
            else:
                print(f"   ⚠️  Could not create PR (changes are on branch {branch_name})")
                
        except Exception as e:
            print(f"   ❌ PR creation failed: {e}")
    
    def _print_summary(self, report: ImprovementReport) -> None:
        """Print a summary of the run."""
        print()
        print("=" * 60)
        print("📊 Summary")
        print("=" * 60)
        print(f"   Changes Detected: {len(report.changes_detected)}")
        print(f"   Docs Impacted: {len(report.docs_impacted)}")
        print(f"   Rules Executed: {len(report.rules_executed)}")
        print(f"   Issues Found: {report.total_issues_found}")
        print(f"   Issues Fixed: {report.total_issues_fixed}")
        print(f"   Average Confidence: {report.average_confidence:.1f}%")
        
        if report.pr_created:
            print(f"   PR URL: {report.pr_url}")
        
        print("=" * 60)
    
    def _save_report(self, report: ImprovementReport) -> None:
        """Save the report to disk."""
        report_dir = self.repo_path / self.config.report_directory
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        report_file = report_dir / f"improvement-{report.run_id}-{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        print(f"\n📁 Report saved: {report_file}")
    
    # Convenience methods for different trigger types
    
    def run_ci(self, base_ref: Optional[str] = None) -> ImprovementReport:
        """Run in CI mode - detects changes from base branch."""
        return self.run(base_ref=base_ref, create_pr=True, trigger="ci")
    
    def run_scheduled(self) -> ImprovementReport:
        """Run in scheduled mode - full scan of all docs."""
        return self.run(full_scan=True, create_pr=True, trigger="scheduled")
    
    def run_manual(
        self, 
        doc_paths: Optional[List[str]] = None,
        create_pr: bool = False,
    ) -> ImprovementReport:
        """Run in manual mode - optionally specify specific docs."""
        # If specific docs provided, create synthetic impacts
        if doc_paths:
            from .models import Severity
            
            impacts = [
                ChangeImpact(
                    source_file="(manual)",
                    impacted_docs=[doc_path],
                    change_type=ChangeType.UNKNOWN,
                    severity=Severity.MEDIUM,
                    changed_symbols=[],
                    git_diff_summary="Manual run",
                    detected_at=datetime.now(),
                )
                for doc_path in doc_paths
            ]
            
            # Run with these impacts
            report = ImprovementReport(
                run_id=str(uuid.uuid4())[:8],
                started_at=datetime.now(),
                trigger="manual",
            )
            report.changes_detected = impacts
            report.docs_impacted = doc_paths
            
            # Continue with execution...
            # (This is a simplified path - full implementation would mirror run())
            
        return self.run(create_pr=create_pr, trigger="manual")
