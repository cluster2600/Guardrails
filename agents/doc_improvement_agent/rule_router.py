"""
Rule Router - Intelligently selects and routes to appropriate documentation rules.

This module determines which rules to apply based on the detected changes,
ensuring required rules (audit-deep, polish) always run, while conditionally
adding specialized rules based on change type.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Set
import json

from .models import ChangeImpact, ChangeType, RuleCategory
from .config import AgentConfig


@dataclass
class RuleSpec:
    """Specification for a documentation rule."""
    name: str
    path: str
    category: RuleCategory
    description: str = ""
    shortcut: str = ""
    priority: int = 50  # 0-100, higher = run earlier


class RuleRouter:
    """Routes documentation changes to appropriate rules."""
    
    # Priority for rule execution (lower = earlier)
    RULE_PRIORITIES = {
        "validation/docs-drift": 10,           # Check drift first
        "validation/docs-code-example-audit": 20,
        "validation/docs-audit-enhanced": 30,  # Core audit
        "validation/docs-audit": 35,
        "validation/docs-confidence-scoring": 40,
        "transformation/docs-polish": 50,      # Polish after validation
        "validation/docs-readability-checker": 60,
        "transformation/docs-modularize-content": 70,
    }
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.rules_path = config.repo_path / config.rules_directory
        self._available_rules: Optional[Dict[str, RuleSpec]] = None
    
    @property
    def available_rules(self) -> Dict[str, RuleSpec]:
        """Lazily load and cache available rules."""
        if self._available_rules is None:
            self._available_rules = self._discover_rules()
        return self._available_rules
    
    def _discover_rules(self) -> Dict[str, RuleSpec]:
        """Discover all available rules in the rules directory."""
        rules = {}
        
        if not self.rules_path.exists():
            return rules
        
        for category_dir in self.rules_path.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            try:
                category = RuleCategory(category_name)
            except ValueError:
                # Skip non-standard directories
                if category_name in ["validation", "transformation", "content", "workflows"]:
                    category = RuleCategory.VALIDATION  # Default
                else:
                    continue
            
            for rule_dir in category_dir.iterdir():
                if not rule_dir.is_dir():
                    continue
                
                rule_file = rule_dir / "RULE.md"
                if rule_file.exists():
                    rule_name = f"{category_name}/{rule_dir.name}"
                    description = self._extract_rule_description(rule_file)
                    
                    rules[rule_name] = RuleSpec(
                        name=rule_dir.name,
                        path=rule_name,
                        category=category,
                        description=description,
                        priority=self.RULE_PRIORITIES.get(rule_name, 50),
                    )
        
        return rules
    
    def _extract_rule_description(self, rule_file: Path) -> str:
        """Extract description from rule frontmatter."""
        try:
            content = rule_file.read_text(encoding='utf-8')
            
            # Check for YAML frontmatter
            if content.startswith('---'):
                end_idx = content.find('---', 3)
                if end_idx > 0:
                    frontmatter = content[3:end_idx]
                    for line in frontmatter.split('\n'):
                        if line.startswith('description:'):
                            return line.split(':', 1)[1].strip()
            
            # Fall back to first heading or paragraph
            lines = content.split('\n')
            for line in lines:
                if line.startswith('# '):
                    return line[2:].strip()
            
            return ""
            
        except Exception:
            return ""
    
    def select_rules(self, impact: ChangeImpact) -> List[RuleSpec]:
        """
        Select appropriate rules for a detected change impact.
        
        Always includes required rules (audit-deep, polish).
        Conditionally adds rules based on change type.
        
        Args:
            impact: The detected change impact
            
        Returns:
            List of RuleSpec objects, sorted by priority
        """
        selected: Set[str] = set()
        
        # 1. Always add required rules
        for rule_path in self.config.required_rules:
            if rule_path in self.available_rules:
                selected.add(rule_path)
        
        # 2. Add conditional rules based on change type
        change_type_key = impact.change_type.value
        if change_type_key in self.config.conditional_rules:
            for rule_path in self.config.conditional_rules[change_type_key]:
                if rule_path in self.available_rules:
                    selected.add(rule_path)
        
        # 3. Add drift detection for high-severity changes
        if impact.severity.value in ["critical", "high"]:
            if "validation/docs-drift" in self.available_rules:
                selected.add("validation/docs-drift")
        
        # 4. Convert to RuleSpec list and sort by priority
        rules = [self.available_rules[path] for path in selected]
        rules.sort(key=lambda r: r.priority)
        
        return rules
    
    def select_rules_for_docs(
        self, 
        doc_paths: List[str], 
        change_type: Optional[ChangeType] = None
    ) -> List[RuleSpec]:
        """
        Select rules for a list of documentation files.
        
        This is a simplified version that just returns required + conditional rules.
        """
        selected: Set[str] = set()
        
        # Always add required rules
        for rule_path in self.config.required_rules:
            if rule_path in self.available_rules:
                selected.add(rule_path)
        
        # Add conditional rules if change type specified
        if change_type:
            change_type_key = change_type.value
            if change_type_key in self.config.conditional_rules:
                for rule_path in self.config.conditional_rules[change_type_key]:
                    if rule_path in self.available_rules:
                        selected.add(rule_path)
        
        # Convert to RuleSpec list and sort by priority
        rules = [self.available_rules[path] for path in selected]
        rules.sort(key=lambda r: r.priority)
        
        return rules
    
    def get_rule_content(self, rule_spec: RuleSpec) -> str:
        """Get the full content of a rule file."""
        rule_file = self.rules_path / rule_spec.path / "RULE.md"
        
        if rule_file.exists():
            return rule_file.read_text(encoding='utf-8')
        
        return ""
    
    def get_rule_by_shortcut(self, shortcut: str) -> Optional[RuleSpec]:
        """Find a rule by its shortcut (e.g., '::a-2' -> audit-enhanced)."""
        shortcut_mapping = {
            "::a": "validation/docs-audit",
            "::a-2": "validation/docs-audit-enhanced",
            "::p": "transformation/docs-polish",
            "::drift": "validation/docs-drift",
        }
        
        rule_path = shortcut_mapping.get(shortcut)
        if rule_path and rule_path in self.available_rules:
            return self.available_rules[rule_path]
        
        return None
    
    def explain_selection(self, impact: ChangeImpact, rules: List[RuleSpec]) -> str:
        """Generate an explanation of why these rules were selected."""
        lines = [
            f"## Rule Selection for `{impact.source_file}`",
            f"",
            f"**Change Type**: {impact.change_type.value}",
            f"**Severity**: {impact.severity.value}",
            f"**Impacted Docs**: {len(impact.impacted_docs)} files",
            f"",
            f"### Selected Rules ({len(rules)})",
            f"",
        ]
        
        for i, rule in enumerate(rules, 1):
            lines.append(f"{i}. **{rule.path}** (Priority: {rule.priority})")
            if rule.description:
                lines.append(f"   - {rule.description}")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_execution_plan(
        self, 
        impacts: List[ChangeImpact]
    ) -> Dict[str, List[RuleSpec]]:
        """
        Generate a complete execution plan for all detected impacts.
        
        Returns:
            Dict mapping doc_path -> list of rules to apply
        """
        plan: Dict[str, List[RuleSpec]] = {}
        
        for impact in impacts:
            rules = self.select_rules(impact)
            
            for doc_path in impact.impacted_docs:
                if doc_path not in plan:
                    plan[doc_path] = []
                
                # Add rules that aren't already in the plan for this doc
                existing_paths = {r.path for r in plan[doc_path]}
                for rule in rules:
                    if rule.path not in existing_paths:
                        plan[doc_path].append(rule)
        
        # Sort rules by priority for each doc
        for doc_path in plan:
            plan[doc_path].sort(key=lambda r: r.priority)
        
        return plan
