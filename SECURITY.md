# Security Policy

## Reporting a Vulnerability

If you've found a security vulnerability in OpenBias, please DO NOT report it via a public GitHub issue. Instead, email us privately.

We will acknowledge your email within 48 hours and provide a timeline for a fix and public disclosure.

## Scope

The following are considered in-scope security issues:
- Bypassing policy evaluation through prompt injection or specific token sequences.
- Exposure of sensitive configuration or API keys through logs or headers.
- Remote code execution in the proxy layer or policy engines.
- Significant denial of service vulnerabilities.

## API Key Safety

OpenBias never stores or transmits your API keys. They are used only for authentication with upstream LLM providers (via LiteLLM).

**Important**:
- Never commit your `.env` file to version control.
- Use `.env.example` as a template for your local configuration.
- We recommend using environment-specific API keys with limited permissions.
- If you accidentally commit a key, rotate it immediately and use `git-filter-repo` or `BFG Repo-Cleaner` to remove it from history.
