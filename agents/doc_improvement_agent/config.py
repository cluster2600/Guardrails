"""
Configuration for the Documentation Self-Improvement Agent.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import yaml


@dataclass
class CodeToDocsMapping:
    """Mapping from code patterns to documentation locations."""
    pattern: str
    impacts: List[str]
    severity: str = "medium"
    change_type: str = "unknown"


@dataclass
class AgentConfig:
    """Configuration for the documentation improvement agent."""
    
    # Repository settings
    repo_path: Path = field(default_factory=lambda: Path.cwd())
    docs_directories: List[str] = field(default_factory=lambda: ["docs/"])
    rules_directory: str = ".cursor/rules"
    
    # Git settings
    default_base_ref: str = "main"
    branch_prefix: str = "docs/auto-improvement"
    
    # Authentication (from environment)
    github_token: Optional[str] = None
    gitlab_token: Optional[str] = None
    
    # Code-to-docs mappings
    code_to_docs_mappings: List[CodeToDocsMapping] = field(default_factory=list)
    
    # Customer-facing patterns (files that affect user experience)
    customer_facing_patterns: List[str] = field(default_factory=lambda: [
        "src/api/**/*.py",
        "src/public/**/*",
        "examples/**/*",
        "configs/**/*.yaml",
        "configs/**/*.yml",
        "**/schema*.py",
        "**/models.py",
        "**/cli.py",
        "**/endpoints.py",
    ])
    
    # Required rules (always run)
    required_rules: List[str] = field(default_factory=lambda: [
        "validation/docs-audit-enhanced",  # audit-deep
        "transformation/docs-polish",       # polish
    ])
    
    # Conditional rules (run based on change type)
    conditional_rules: Dict[str, List[str]] = field(default_factory=lambda: {
        "api": ["validation/docs-code-example-audit"],
        "schema": ["validation/docs-drift"],
        "config": ["validation/docs-drift"],
        "example": ["validation/docs-code-example-audit"],
    })
    
    # Thresholds
    min_confidence_for_pr: float = 70.0
    max_files_per_pr: int = 10
    
    # Output settings
    report_directory: str = ".dori/reports"
    state_directory: str = ".dori/state"
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "AgentConfig":
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        if 'repo_path' in data:
            config.repo_path = Path(data['repo_path'])
        if 'docs_directories' in data:
            config.docs_directories = data['docs_directories']
        if 'rules_directory' in data:
            config.rules_directory = data['rules_directory']
        if 'default_base_ref' in data:
            config.default_base_ref = data['default_base_ref']
        if 'branch_prefix' in data:
            config.branch_prefix = data['branch_prefix']
        if 'customer_facing_patterns' in data:
            config.customer_facing_patterns = data['customer_facing_patterns']
        if 'required_rules' in data:
            config.required_rules = data['required_rules']
        if 'conditional_rules' in data:
            config.conditional_rules = data['conditional_rules']
        if 'min_confidence_for_pr' in data:
            config.min_confidence_for_pr = data['min_confidence_for_pr']
        
        # Load code-to-docs mappings
        if 'code_to_docs_mappings' in data:
            config.code_to_docs_mappings = [
                CodeToDocsMapping(**m) for m in data['code_to_docs_mappings']
            ]
        
        # Load tokens from environment
        config.github_token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
        config.gitlab_token = os.environ.get('GITLAB_TOKEN') or os.environ.get('CI_JOB_TOKEN')
        
        return config
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load configuration from environment variables."""
        config = cls()
        config.github_token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
        config.gitlab_token = os.environ.get('GITLAB_TOKEN') or os.environ.get('CI_JOB_TOKEN')
        
        if env_base_ref := os.environ.get('DOC_AGENT_BASE_REF'):
            config.default_base_ref = env_base_ref
        
        return config
    
    def to_yaml(self, output_path: Path) -> None:
        """Save configuration to a YAML file."""
        data = {
            'docs_directories': self.docs_directories,
            'rules_directory': self.rules_directory,
            'default_base_ref': self.default_base_ref,
            'branch_prefix': self.branch_prefix,
            'customer_facing_patterns': self.customer_facing_patterns,
            'required_rules': self.required_rules,
            'conditional_rules': self.conditional_rules,
            'min_confidence_for_pr': self.min_confidence_for_pr,
            'code_to_docs_mappings': [
                {
                    'pattern': m.pattern,
                    'impacts': m.impacts,
                    'severity': m.severity,
                    'change_type': m.change_type,
                }
                for m in self.code_to_docs_mappings
            ],
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Default configuration for common project structures
DEFAULT_CODE_TO_DOCS_MAPPINGS = [
    CodeToDocsMapping(
        pattern="src/api/**/*.py",
        impacts=["docs/api-reference/**", "docs/getting-started/**"],
        severity="high",
        change_type="api",
    ),
    CodeToDocsMapping(
        pattern="src/config*.py",
        impacts=["docs/configuration/**", "docs/reference/**"],
        severity="medium",
        change_type="config",
    ),
    CodeToDocsMapping(
        pattern="examples/**/*.py",
        impacts=["docs/tutorials/**", "docs/getting-started/**"],
        severity="high",
        change_type="example",
    ),
    CodeToDocsMapping(
        pattern="**/schema*.py",
        impacts=["docs/api-reference/**"],
        severity="high",
        change_type="schema",
    ),
    CodeToDocsMapping(
        pattern="**/models.py",
        impacts=["docs/api-reference/**", "docs/concepts/**"],
        severity="medium",
        change_type="model",
    ),
    CodeToDocsMapping(
        pattern="**/cli.py",
        impacts=["docs/cli-reference/**", "docs/getting-started/**"],
        severity="high",
        change_type="cli",
    ),
]
