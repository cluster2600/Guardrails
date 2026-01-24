# Documentation Self-Improvement Agent for NeMo Guardrails Library

An intelligent agent that automatically detects code changes affecting documentation, runs appropriate validation and improvement rules, and creates PRs with fixes.

Adapted for the NeMo Guardrails repository structure.

## Features

- **Automatic Change Detection**: Analyzes git diffs to identify code changes that impact documentation
- **Intelligent Rule Routing**: Selects appropriate rules based on change type (API, config, schema, etc.)
- **Required Rules**: Always runs `audit-deep` and `polish` for quality assurance
- **Multi-Trigger Support**: CI/CD, scheduled, manual, or LLM-driven execution
- **PR Automation**: Creates pull requests with detailed summaries of improvements

## Quick Start

### Manual Execution

```bash
# Detect changes from main branch
python -m agents.doc_improvement_agent --base-ref main

# Detect changes and create a PR
python -m agents.doc_improvement_agent --base-ref main --create-pr

# Full scan of all documentation
python -m agents.doc_improvement_agent --full-scan

# Dry run (show what would be done)
python -m agents.doc_improvement_agent --dry-run
```

### In Python

```python
from agents.doc_improvement_agent import DocImprovementAgent

# Create agent
agent = DocImprovementAgent(repo_path=".")

# Run with options
report = agent.run(
    base_ref="main",
    create_pr=True,
    trigger="manual"
)

print(f"Issues found: {report.total_issues_found}")
print(f"Issues fixed: {report.total_issues_fixed}")
if report.pr_url:
    print(f"PR created: {report.pr_url}")
```

## Authentication

### GitHub

Set one of these environment variables:

- `GITHUB_TOKEN`: Personal Access Token with `repo` scope
- `GH_TOKEN`: Same as above (used by gh CLI)

The agent uses `gh` CLI for PR creation. Ensure it's installed and authenticated:

```bash
gh auth login
```

### GitLab

Set one of these environment variables:

- `GITLAB_TOKEN`: Personal Access Token with `api` scope
- `CI_JOB_TOKEN`: Automatically available in GitLab CI

The agent uses `glab` CLI for MR creation. Ensure it's installed:

```bash
glab auth login
```

### CI/CD Setup

#### GitLab CI Variables

Add these in **Settings → CI/CD → Variables**:

| Variable | Description | Scope |
|----------|-------------|-------|
| `GITHUB_TOKEN` | GitHub PAT (if pushing to GitHub) | Protected |
| `OPENAI_API_KEY` | For LLM-based rule execution | Protected |

The `CI_JOB_TOKEN` is automatically available for GitLab operations.

#### GitHub Actions

```yaml
- name: Run Doc Improvement Agent
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    python -m agents.doc_improvement_agent --ci
```

## Configuration

Create a `doc-agent.yaml` in your repository root:

```yaml
# Documentation directories to scan
docs_directories:
  - "docs/"
  - "examples/"

# Required rules (always run)
required_rules:
  - "validation/docs-audit-enhanced"
  - "transformation/docs-polish"

# Code-to-docs mapping
code_to_docs_mappings:
  - pattern: "src/api/**/*.py"
    impacts:
      - "docs/api-reference/**"
    severity: "high"
    change_type: "api"
```

See the full configuration schema in `config.py`.

## Trigger Modes

### 1. CI/CD (on merge)

Runs automatically when code is merged to main:

```yaml
# In .gitlab-ci.yml
doc-improvement:on-merge:
  script:
    - python -m agents.doc_improvement_agent --ci
  rules:
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
```

### 2. Scheduled (weekly scan)

Runs on a schedule to catch drift:

```yaml
doc-improvement:scheduled:
  script:
    - python -m agents.doc_improvement_agent --scheduled
  rules:
    - if: $CI_PIPELINE_SOURCE == "schedule"
```

### 3. User Command (Cursor)

Use the `::improve` command in Cursor:

```
::improve              # Check recent changes
::improve --full       # Full documentation scan
::improve --pr         # Create PR with fixes
```

### 4. LLM-Driven

Integrate with LLM for intelligent execution:

```python
agent = DocImprovementAgent(
    repo_path=".",
    llm_executor=my_llm_function  # Custom LLM integration
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Documentation Self-Improvement Agent          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ChangeDetector    →   RuleRouter    →   Agent (executor)   │
│  (git diff)            (select rules)    (run rules)        │
│                                                             │
│                        ↓                                    │
│                                                             │
│                   PRGenerator                               │
│                   (create PR)                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| `ChangeDetector` | Analyzes git diffs, maps code changes to doc impacts |
| `RuleRouter` | Selects rules based on change type and severity |
| `DocImprovementAgent` | Orchestrates detection, execution, and PR creation |
| `PRGenerator` | Creates branches, commits, and pull requests |

## Rules Applied

### Required (Always Run)

- **validation/docs-audit-enhanced**: Deep audit with confidence scoring and triangulation
- **transformation/docs-polish**: Style guide compliance and clarity improvements

### Conditional (Based on Change Type)

| Change Type | Additional Rules |
|-------------|------------------|
| API | `docs-code-example-audit` |
| Schema | `docs-drift` |
| Config | `docs-drift` |
| Example | `docs-code-example-audit` |

## Reports

Reports are saved to `.dori/reports/` in JSON format:

```json
{
  "run_id": "abc123",
  "trigger": "ci",
  "changes_detected": [...],
  "docs_impacted": [...],
  "rules_executed": [...],
  "total_issues_found": 5,
  "total_issues_fixed": 3,
  "average_confidence": 87.5,
  "pr_url": "https://..."
}
```

## Troubleshooting

### No changes detected

1. Ensure your branch has commits compared to base ref
2. Check that changed files match `customer_facing_patterns`
3. Verify `docs_directories` are correct

### PR creation fails

1. Check authentication (run `gh auth status` or `glab auth status`)
2. Ensure you have write permissions to the repository
3. Check for branch protection rules

### Rules not executing

1. Verify rules exist in `.cursor/rules/`
2. Check rule paths in configuration
3. For full LLM execution, provide `llm_executor`

## Development

### Running Tests

```bash
pytest agents/doc_improvement_agent/tests/
```

### Adding New Rules

1. Add rule mapping to `conditional_rules` in config
2. Update `RuleRouter.RULE_PRIORITIES` if needed
3. Add script mapping in `agent._execute_deterministic_rule` if applicable

## License

See project LICENSE file.
