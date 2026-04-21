# Security Policy

## Reporting a Vulnerability

Email **studio@newmediastudio.com** with:

- A description of the issue
- Steps to reproduce
- Affected versions
- Potential impact
- Any suggested fix

Do **not** open a public GitHub issue for security vulnerabilities.

We acknowledge reports within 72 hours and coordinate disclosure from there.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.9.x   | Yes       |
| < 1.9   | No        |

Upgrade to the latest minor release for security fixes.

## Scope

Hush Engine runs locally. The only network calls are:

- First-run model downloads from HuggingFace and spaCy
- Optional API calls to Anthropic and Google when you enable those clients in the LLM comparison benchmark

In scope:
- PII detection bypass (the engine fails to detect a value it should)
- Code execution, path traversal, or resource exhaustion from crafted input
- Runtime vulnerabilities in dependencies that affect Hush

Out of scope:
- Misdetections where regular text is flagged as PII. Open a regular GitHub issue for those.
- Upstream vulnerabilities in Presidio, spaCy, or other dependencies. Report those to their maintainers; we upgrade when fixes land.

## Safe Use

Hush reads user-provided files (images, PDFs, spreadsheets). Treat any input parsed by the engine as untrusted. Run redaction in a sandbox (container, nsjail, macOS App Sandbox) when processing content from unknown sources.
