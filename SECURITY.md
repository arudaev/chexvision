# Security Policy

## Supported Versions

Only the latest commit on `main` is actively maintained.

| Branch / Version | Supported |
|------------------|-----------|
| `main` (latest)  | ✅        |
| Any older commit | ❌        |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report vulnerabilities through [GitHub Private Security Advisories](https://github.com/arudaev/chexvision/security/advisories/new).
Your report will be visible only to the repository maintainers until it is resolved and publicly disclosed.

### What to include

- A clear description of the vulnerability and its potential impact
- Steps to reproduce or a minimal proof-of-concept
- Affected component (e.g. `app/app.py`, `src/utils/hub.py`, Kaggle kernel scripts)
- Any suggested mitigation, if you have one

## Response Timeline

| Stage | Target |
|-------|--------|
| Acknowledgement | Within **72 hours** |
| Initial assessment | Within **7 days** |
| Fix or mitigation | Within **30 days** (critical issues prioritised) |

## Scope

### In scope

- Credential or token exposure (HF token, Kaggle API token, GitHub token)
- Unsafe model loading or arbitrary code execution via checkpoint files
- Vulnerabilities in the Streamlit demo (`app/`) that could affect users
- Insecure handling of uploaded medical images

### Out of scope

- Model accuracy, bias, or clinical performance — this is a university research project and the demo is **not** intended for medical diagnosis
- Vulnerabilities in third-party dependencies (report those upstream)
- Issues requiring physical access to a machine

## Disclosure Policy

Once a fix is merged into `main`, the advisory will be published publicly. Credit will be given to the reporter unless they prefer to remain anonymous.
