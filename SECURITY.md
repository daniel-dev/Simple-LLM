# Security Policy

## Supported Versions

We actively support the following versions of LLM Factory:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | ✅ Yes             |
| < 1.0   | ❌ No              |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### 🔒 For Security Issues

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please:

1. **Email us directly** at: [security@your-domain.com] (replace with actual email)
2. **Include the following information:**
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact
   - Any suggested fixes (if you have them)

### 📧 Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 1 week
- **Resolution Timeline**: Varies based on severity

### 🛡️ Security Best Practices

When using LLM Factory:

1. **Keep dependencies updated**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Use virtual environments**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Don't commit sensitive data**
   - API keys
   - Personal data
   - Model weights (use Git LFS)

4. **Validate input data**
   - Sanitize file uploads
   - Validate model inputs
   - Check data sources

5. **Use HTTPS in production**
   - Enable SSL/TLS
   - Use secure headers
   - Validate certificates

### 🚨 Known Security Considerations

- **Model Security**: Trained models may contain sensitive information from training data
- **Data Privacy**: Ensure compliance with data protection regulations (GDPR, CCPA, etc.)
- **Dependency Security**: Regularly audit dependencies for known vulnerabilities
- **Input Validation**: Always validate and sanitize user inputs

### 🔍 Security Scanning

We use automated security scanning:

- **Dependencies**: Scanned with `pip-audit`
- **Code**: Static analysis with security tools
- **CI/CD**: Security checks in GitHub Actions

### 📝 Disclosure Policy

- We will acknowledge receipt of vulnerability reports
- We will provide regular updates on investigation progress
- We will publicly disclose vulnerabilities after fixes are deployed
- We will credit security researchers (with permission)

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Guidelines](https://python.org/dev/security/)
- [GitHub Security Best Practices](https://docs.github.com/en/code-security)

Thank you for helping keep LLM Factory secure! 🛡️
