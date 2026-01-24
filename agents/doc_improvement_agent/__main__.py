#!/usr/bin/env python3
"""
Documentation Self-Improvement Agent - CLI Entry Point

Usage:
    python -m agents.doc_improvement_agent [options]
    
Examples:
    # Detect changes from main and report
    python -m agents.doc_improvement_agent --base-ref main
    
    # Detect changes and create PR
    python -m agents.doc_improvement_agent --base-ref main --create-pr
    
    # Full scan of all documentation
    python -m agents.doc_improvement_agent --full-scan --create-pr
    
    # Dry run (no changes)
    python -m agents.doc_improvement_agent --dry-run
    
    # Specify specific docs to process
    python -m agents.doc_improvement_agent --docs docs/getting-started.md docs/api.md
"""

import argparse
import sys
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Documentation Self-Improvement Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Mode options
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--ci",
        action="store_true",
        help="Run in CI mode (detect changes, create PR)",
    )
    mode_group.add_argument(
        "--scheduled",
        action="store_true", 
        help="Run in scheduled mode (full scan, create PR)",
    )
    mode_group.add_argument(
        "--full-scan",
        action="store_true",
        help="Scan all documentation files",
    )
    
    # Git options
    parser.add_argument(
        "--base-ref",
        default="main",
        help="Git reference to compare against (default: main)",
    )
    parser.add_argument(
        "--repo-path",
        default=".",
        help="Path to repository root (default: current directory)",
    )
    
    # Action options
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create a pull request with improvements",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    
    # Specific docs
    parser.add_argument(
        "--docs",
        nargs="+",
        help="Specific documentation files to process",
    )
    
    # Config
    parser.add_argument(
        "--config",
        help="Path to configuration file",
    )
    
    # Output options
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    
    # Version
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    
    args = parser.parse_args()
    
    # Import agent components
    from .agent import DocImprovementAgent
    from .config import AgentConfig
    
    # Load configuration
    if args.config:
        config = AgentConfig.from_yaml(Path(args.config))
    else:
        config = AgentConfig.from_env()
    
    # Create agent
    agent = DocImprovementAgent(
        repo_path=args.repo_path,
        config=config,
    )
    
    # Determine run mode
    if args.ci:
        report = agent.run_ci(base_ref=args.base_ref)
    elif args.scheduled:
        report = agent.run_scheduled()
    elif args.docs:
        report = agent.run_manual(doc_paths=args.docs, create_pr=args.create_pr)
    else:
        report = agent.run(
            base_ref=args.base_ref,
            create_pr=args.create_pr,
            full_scan=args.full_scan,
            dry_run=args.dry_run,
            trigger="manual",
        )
    
    # Output
    if args.json:
        import json
        print(json.dumps(report.to_dict(), indent=2))
    
    # Return appropriate exit code
    if report.pr_created or not args.create_pr:
        sys.exit(0)
    else:
        # PR was requested but not created
        sys.exit(1)


if __name__ == "__main__":
    main()
