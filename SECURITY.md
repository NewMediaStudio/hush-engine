# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Hush Engine, please report it **privately**.

**Do not** open a public GitHub issue for security vulnerabilities.

### How to report

Email: **studio@newmediastudio.com**

Include:
- A description of the vulnerability
- Steps to reproduce
- Affected versions (if known)
- Potential impact
- Any suggested remediation

We will acknowledge your report within 72 hours and work with you on a coordinated disclosure timeline.

## Supported Versions

Only the latest minor version receives security updates. Upgrade to the newest release to stay protected.

| Version | Supported |
|---------|-----------|
| 1.9.x   | ✅        |
| < 1.9   | ❌        |

## Scope

Hush Engine is a **local-first** library — it does not make network calls except for optional model downloads on first use (from HuggingFace / spaCy) and optional API calls to Anthropic/Google when users explicitly enable those clients in the LLM comparison benchmark.

In scope:
- PII detection bypass (an intended-PII value the engine fails to detect)
- Code execution, path traversal, or resource exhaustion via crafted input
- Dependency vulnerabilities that affect Hush Engine at runtime

Out of scope:
- Misdetections ("false positive" — regular words flagged as PII). Report via normal GitHub issues.
- Upstream vulnerabilities in Presidio, spaCy, or other dependencies (report those to their maintainers; we will upgrade when fixes are released).

## Safe Use

Hush Engine reads user-provided files (images, PDFs, spreadsheets). Treat any file parsed by the engine as untrusted input. We recommend running redaction in a sandbox (container, `nsjail`, macOS App Sandbox) when processing content from unknown sources.
