# Security Policy

## Overview

The ANCHURUS project takes security seriously.

ANCHURUS is an experimental lightweight monocular depth estimation project. This policy describes how to report security vulnerabilities and how security issues are handled.

We appreciate responsible security research and encourage anyone who discovers a potential vulnerability to report it.

## Supported Versions

Security updates are currently provided for the latest maintained version of ANCHURUS.

| Version | Supported |
|---|---|
| Latest release | Yes |
| Development branch | Best effort |
| Older versions | No |

Support status may change as the project evolves.

## Reporting a Vulnerability

**Please do not publicly disclose security vulnerabilities through GitHub Issues or Pull Requests before they have been reviewed.**

If you discover a security vulnerability, report it privately through GitHub's private vulnerability reporting feature, if enabled for this repository.

Repository:

https://github.com/Davitestro/ANCHURUS

If private vulnerability reporting is unavailable, contact the repository maintainer directly through their GitHub profile.

### What to Include

Please provide as much information as possible:

- A clear description of the vulnerability.
- Steps to reproduce the issue.
- Potential security impact.
- Affected versions or components.
- Proof of concept, if available.
- Suggested mitigation, if known.

Please avoid including sensitive personal information or confidential data in your report.

## Security Scope

The following areas are particularly relevant to ANCHURUS.

### 1. Model Files and Checkpoints

Model checkpoints and externally supplied weights may contain unsafe serialized objects.

- Do not load untrusted checkpoint files.
- Prefer safer weight formats where practical.
- Treat externally downloaded model files as untrusted input.
- Validate model files before integrating them into the project.

### 2. Dataset Security

Datasets obtained from external sources may contain unexpected or malicious files.

- Verify dataset sources.
- Avoid executing scripts distributed with untrusted datasets.
- Validate input files before processing.
- Do not commit private or sensitive datasets.

### 3. Dependencies

Third-party libraries may introduce security vulnerabilities.

- Keep dependencies reasonably up to date.
- Review new dependencies before introducing them.
- Avoid unnecessary packages.
- Investigate reported vulnerabilities affecting project dependencies.

### 4. Inference and Input Handling

Unexpected inputs may cause crashes, excessive resource consumption, or unexpected behavior.

Contributors should consider:

- Invalid image formats.
- Extremely large images.
- Corrupted input files.
- Unexpected tensor dimensions.
- Excessive memory consumption.
- Unsafe file paths.

### 5. Credentials and Sensitive Information

Never commit:

- API keys.
- Passwords.
- Access tokens.
- Private SSH keys.
- Personal information.
- Local environment configuration containing secrets.

## Response Process

After receiving a vulnerability report, the maintainer will:

1. Review the report and attempt to reproduce the issue.
2. Assess its potential impact.
3. Determine affected versions and components.
4. Develop or coordinate an appropriate fix.
5. Publish security updates and relevant advisories when appropriate.

Response times depend on the severity and complexity of the reported issue.

## Responsible Disclosure

Please allow the maintainer reasonable time to investigate and address reported vulnerabilities before publicly disclosing them.

We appreciate responsible disclosure and cooperation from security researchers.

## Scope Limitations

This policy covers security vulnerabilities directly related to ANCHURUS and its official repository.

It does not guarantee security audits, continuous monitoring, or a specific response time.

## Acknowledgments

We appreciate researchers and contributors who help identify and responsibly report security issues.

Thank you for helping keep ANCHURUS and its users safe.

---

**Project:** [Davitestro/ANCHURUS](https://github.com/Davitestro/ANCHURUS)
