## Summary
<!-- Brief description of changes.  Focus on *why* the change is needed, not
     just *what* was changed — reviewers can read the diff for the "what". -->

## Type of change
<!-- Tick exactly one.  If a PR spans multiple categories (e.g. a bug fix that
     also adds a test), choose the primary motivation.  This label drives the
     CHANGELOG grouping and release-note categorisation. -->
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Refactoring
- [ ] Documentation
- [ ] CI/infrastructure

## Pre-merge checklist

### Thread safety
<!-- These items guard against regressions introduced by free-threaded Python
     (3.14t / no-GIL builds).  On GIL-enabled builds bare dicts are technically
     safe for single-operation access, but they break silently under 3.14t
     concurrency — hence the blanket requirement for ThreadSafeDict/Cache. -->
- [ ] All new shared mutable state uses `ThreadSafeDict` or `ThreadSafeCache`
- [ ] All caches are bounded (maxsize parameter)
- [ ] No bare `dict` for concurrently-accessed registries

### Type safety
<!-- These catch real bugs encountered during the thread-safety work:
     - `type(x) is dict` fails for ThreadSafeDict (a dict subclass);
       `isinstance` is the correct check.
     - Omitting parentheses on methods returns the bound method object rather
       than calling it — a subtle bug that passes truthiness checks silently.
     - Pydantic v2 ignores @property during serialisation; @computed_field
       ensures the value appears in .model_dump() / JSON output. -->
- [ ] No `type(...) is dict` — use `isinstance` for subclass compatability
- [ ] Method calls use parentheses (e.g. `cache.stats()` not `cache.stats`)
- [ ] `@property` on Pydantic models uses `@computed_field` if serialisation is needed

### Configuration
<!-- Guardrails merges multiple YAML configs via _join_config().  Any new
     top-level field that is not registered in `additional_fields` will be
     silently dropped during the merge — a common source of "config not working"
     reports.  The env-var and dependency items are standard hygiene. -->
- [ ] New config fields added to `_join_config()` `additional_fields`
- [ ] Environment variable parsing has error handling with fallback defaults
- [ ] New dependencies added to `pyproject.toml`

### Testing
<!-- "Tests call production code paths" means: do not copy-paste internal logic
     into the test and assert against the copy.  Instead, import and invoke the
     real function.  This avoids tests that pass whilst the production code has
     diverged.  The stress-test item ensures new thread-safe structures are
     exercised under genuine contention, not just single-threaded happy paths. -->
- [ ] Tests call production code paths, not reimplemented logic
- [ ] Thread safety stress tests added for new concurrent data structures
- [ ] No unused imports (ruff will catch this)

### Documentation
<!-- Sphinx will emit a warning for any .rst/.md file not referenced in a
     toctree directive.  The `:orphan:` metadata suppresses this for standalone
     documents (e.g. research notes, one-off guides) that should not appear in
     the navigation sidebar. -->
- [ ] Comments accurately describe code behaviour
- [ ] New docs added to a toctree or marked as `:orphan:`

## Test plan
<!-- How was this tested?  Include: (1) which test commands were run,
     (2) whether manual testing was performed (e.g. against a live LLM endpoint),
     and (3) any benchmark results if the PR touches performance-sensitive paths.
     This section is required — PRs without a test plan will not be merged. -->
