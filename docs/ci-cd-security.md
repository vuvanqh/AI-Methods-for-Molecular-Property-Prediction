# CI/CD and Repository Security

This document describes the repository's initial CI/CD and security baseline. Repository-level settings are based on the accompanying GitHub settings review and the active `main protection` ruleset.
## Continuous Integration

The repository has three GitHub Actions workflows:

| Workflow | Triggers | Current behavior |
| --- | --- | --- |
| `CI` | Pushes and pull requests targeting `main` | Runs backend validation on Python 3.11 and 3.12 and frontend validation on Node.js 20. |
| `CodeQL` | Pushes and pull requests targeting `main`; weekly on Monday| Analyzes Python and TypeScript with the `security-extended` and `security-and-quality` query suites. |
| `Dependency Review` | Pull requests targeting `main` | Rejects dependency changes with known vulnerabilities of moderate severity or higher. |

Backend CI executes Ruff and pytest according to the repository's workflow configuration. Frontend CI installs dependencies and executes configured lint, test, and build scripts. Checks that are conditional on the presence of directories or scripts may be skipped. As the application is developed, the relevant validation steps will become mandatory to ensure that missing configurations do not silently bypass CI validation.

The active branch ruleset currently requires these check names before a merge:
- `Backend CI (Python 3.12)`
- `Frontend CI`
- `Analyze (python)`
- `Analyze (javascript-typescript)`
- `Review Dependencies`

Python 3.11 is tested but is not currently a required status check as the exact version is still to be decided.

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

