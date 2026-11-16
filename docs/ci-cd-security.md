# CI/CD and Repository Security

This document describes the repository's initial CI/CD and security baseline. Repository-level settings are based on the accompanying GitHub settings review and the active `main protection` ruleset.
## Continuous Integration

The repository has three GitHub Actions workflows:

| Workflow | Triggers | Current behavior |
| --- | --- | --- |
| `CI` | Pushes and pull requests targeting `main` | Runs Ruff and pytest for each backend project (`app`, `services/gnn_xai_service`) on its pinned Python version, frontend lint, coverage, and build on Node.js 24, and Playwright E2E tests. |
| `CodeQL` | Pushes and pull requests targeting `main`; weekly on Monday | Analyzes Python and TypeScript with the `security-extended` and `security-and-quality` query suites. |
| `Dependency Review` | Pull requests targeting `main` | Rejects dependency changes with known vulnerabilities of moderate severity or higher. |

Each backend service is its own uv project with its own `pyproject.toml`, `uv.lock`, and Python pin (`.python-version` in the service folder, otherwise the repository root's), so CI runs the same versions as local development. Backend CI installs with `uv sync --locked` and runs Ruff and pytest. Frontend CI runs lint, tests with coverage thresholds, and the build. The Playwright E2E job runs against a mocked API. Some steps are skipped when a project has no tests or no `package.json` yet.

The active branch ruleset currently requires these check names before a merge:
- `Backend CI (app)`
- `Backend CI (services/gnn_xai_service)`
- `Frontend CI`
- `Analyze (python)`
- `Analyze (javascript-typescript)`
- `Review Dependencies`

`Frontend E2E (Playwright)` runs on every pull request but is not a required check.

## Dependency and Security

The repository settings enable the dependency graph, Dependabot vulnerability alerts, malware alerts, Dependabot security updates and grouped security updates. Together, these features report known vulnerable or malicious packages, and propose security fixes. Grouped security updates are a repository setting and are separate from version-update groups in `dependabot.yml`.

Dependabot version updates are configured weekly as follows:

| Ecosystem | Directory | Schedule | Pull request limit | Commit prefix |
| --- | --- | --- | --- | --- |
| GitHub Actions | `/` | Weekly | GitHub default | `ci` with scope |
| npm | `/frontend` | Weekly | 10 | `frontend` with scope |
| pip | `/backend` | Weekly | 10 | `backend` with scope |

Dependabot proposes dependency changes. Dependency Review evaluates the dependency changes in a pull request. Neither controls the application's behavior, so both complement the CI jobs.

## Static Code Analysis

CodeQL provides static application security testing for Python and TypeScript. It analyzes source code for potential vulnerabilities and coding defects using the configured query suites.

CodeQL runs on pushes and pull requests targeting main, as well as on a weekly schedule. The repository requires both successful CodeQL workflow checks and registered code scanning results before merging into protected branches.

A successful workflow check does not necessarily indicate that code scanning results were uploaded and registered. Both requirements must be satisfied independently.

## Secret Protection

Secret scanning and push protection are enabled in the repository settings. Push protection blocks commits containing secret types that GitHub can detect.

***IMPORTANTLY***: These controls reduce accidental credential exposure but cannot detect every secret format. 

## Branch Protection

The active `main protection` ruleset:

- targets `main` and `master`
- requires a pull request
- dismisses stale approvals
- requires all review conversations to be resolved
- requires all status checks to pass 
- requires the branch to be up to date before merging
- blocks branch deletion and force pushes
- requires CodeQL code scanning results

The ruleset permits merge commits, squash merges, and rebases.

