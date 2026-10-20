# CI/CD and Repository Security

This document describes the repository's initial CI/CD and security baseline. File-based configuration was verified against `.github/` on 20 October 2026. Repository-level settings are based on the accompanying GitHub settings review and the active `main protection` ruleset. Settings managed only in GitHub should be rechecked after a ruleset or security-settings change.

## Continuous Integration

The repository has three GitHub Actions workflows:

| Workflow | Triggers | Current behavior |
| --- | --- | --- |
| `CI` | Pushes and pull requests targeting `main` | Runs backend validation on Python 3.11 and 3.12 and frontend validation on Node.js 20. |
| `CodeQL` | Pushes and pull requests targeting `main`; weekly on Monday at 02:30 UTC | Analyzes Python and JavaScript/TypeScript with the `security-extended` and `security-and-quality` query suites. |
| `Dependency Review` | Pull requests targeting `main` | Rejects dependency changes with known vulnerabilities of moderate severity or higher. |

Backend CI runs Ruff and, when `backend/tests/` exists, pytest. It is intentionally a no-op while neither `backend/requirements.txt` nor `backend/pyproject.toml` exists. Frontend CI installs dependencies and runs available lint, test, and build scripts. It is intentionally a no-op until `frontend/package.json` exists. These checks validate code and builds; they do not replace dependency or static security analysis.

The active branch ruleset currently requires these check names before a merge:

- `Backend CI (Python 3.12)`
- `Frontend CI`
- `Analyze (python)`
- `Analyze (javascript-typescript)`
- `Review Dependencies`

Python 3.11 is tested but is not currently a required status check. This provides compatibility feedback without making that matrix entry a merge gate.

## Dependency and Supply-Chain Security

The reviewed repository settings enable the dependency graph, automatic dependency submission, Dependabot vulnerability alerts, malware alerts, Dependabot security updates, grouped security updates, and one Dependabot auto-triage rule. Together, these features inventory dependencies, report known vulnerable or malicious packages, and propose security fixes. Grouped security updates are a repository setting and are separate from version-update groups in `dependabot.yml`.

Dependabot version updates are configured weekly as follows:

| Ecosystem | Directory | Schedule | Pull request limit | Commit prefix | Version-update groups |
| --- | --- | --- | --- | --- | --- |
| GitHub Actions | `/` | Weekly | GitHub default | `ci` with scope | None configured |
| npm | `/frontend` | Weekly | 10 | `frontend` with scope | None configured |
| pip | `/backend` | Weekly | 10 | `backend` with scope | None configured |

Dependabot proposes dependency changes. Dependency Review evaluates the dependency changes in a pull request. Neither control tests application behavior, so both complement the CI jobs.

## Static Code Analysis

CodeQL is configured for Python and JavaScript/TypeScript. Each matrix job first checks whether matching source files exist. If none exist, the job succeeds without initializing CodeQL or uploading an analysis result. This behavior suits the current template, whose `backend/` and `frontend/` directories contain no application source code, but it creates an enforcement distinction:

- A **required CodeQL workflow check** requires the `Analyze (...)` Actions job to complete successfully.
- A **required code scanning result** requires GitHub code scanning to receive and register an analysis result for the protected branch and language.

The active ruleset currently enables both required CodeQL workflow checks and required code scanning results. With no analyzable source, the workflow can pass while no scanning result is uploaded. GitHub may then block merging with:

> Merging is blocked. Waiting for Code Scanning results. Code Scanning may not be configured for the target branch.

The lack of source is a plausible cause in this template, but the precise cause has not been verified. Keep the CodeQL workflow enabled. During the template phase, disable only the ruleset's required **code scanning results** rule if it blocks merges without a valid result. Keep the `Analyze (...)` workflow checks required. Re-enable scanning-result enforcement after application code is added and successful results are visible for `main` and for pull requests.

## Secret Protection

Secret scanning and push protection are enabled in the reviewed repository settings. Push protection blocks commits containing secret types that GitHub can detect. Store repository and environment secrets only where needed, grant the narrowest practical permissions, and prefer environment-scoped credentials for deployment targets.

These controls reduce accidental credential exposure but cannot detect every secret format. Application code must not treat secret scanning as a substitute for credential rotation or appropriate runtime secret storage.

## Branch Protection

The active `main protection` ruleset:

- targets `main` and `master`, with no bypass actors;
- requires a pull request, dismisses stale approvals, and currently requires zero approvals;
- requires all review conversations to be resolved;
- requires the five status checks listed above and requires the branch to be up to date before merging;
- blocks branch deletion and force pushes; and
- currently requires CodeQL code scanning results.

The ruleset permits merge commits, squash merges, and rebases. Review approval requirements can be increased when the project has enough maintainers to support them.

All workflows currently run only for `main`, although the ruleset also targets `master`. If `master` remains a valid merge target, the workflows must also target it or its required checks will not be produced. Otherwise, remove `master` from the ruleset target criteria.

Branch protection coordinates merge policy; it does not perform linting, tests, vulnerability checks, or static analysis itself. It makes results from those separate systems mandatory.

## Additional Security Features

The reviewed repository settings have Copilot Autofix enabled. AI Scan for pull requests and Agentic Autofix are disabled. Private vulnerability reporting is also disabled; enable it if the repository becomes public and accepts confidential reports from external security researchers.

## Template-Phase Follow-Up

When meaningful application code and dependency manifests are introduced:

1. Confirm that backend and frontend jobs execute real lint, test, and build commands rather than their template no-op paths.
2. Confirm that CodeQL uploads Python and JavaScript/TypeScript results on `main` and on pull requests.
3. Re-enable required code scanning results if they were temporarily relaxed.
4. Align the ruleset target branches with workflow trigger branches.
5. Reassess required reviews, test coverage, deployment protection, and dependency grouping based on the implemented technology stack.
