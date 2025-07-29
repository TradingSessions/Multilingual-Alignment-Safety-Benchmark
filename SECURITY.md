# Security Policy

## 🔐 Security Commitment

The MASB-Alt project takes security seriously. As an AI safety evaluation tool, we recognize the importance of maintaining robust security practices to protect user data, API credentials, and evaluation results.

## 🛡️ Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | ✅ Yes             |
| < 1.0   | ❌ No              |

## 📋 Security Considerations

### API Key Security
- **Never commit API keys** to version control
- Store API keys in environment variables or secure configuration files
- Use the provided `.env.example` template for configuration
- Implement key rotation practices for production deployments

### Data Protection
- Evaluation data may contain sensitive information
- Implement appropriate access controls for databases
- Consider data encryption for sensitive evaluation results
- Regular backup with secure storage practices

### Network Security
- Use HTTPS for all API communications
- Implement proper CORS policies for web services
- Consider VPN or private networks for sensitive deployments
- Monitor API usage and implement rate limiting

### Dependency Security
- Regularly update dependencies to patch security vulnerabilities
- Use `pip audit` or similar tools to check for known vulnerabilities
- Review dependency licenses for compliance requirements

## 🚨 Reporting Security Vulnerabilities

We take security vulnerabilities seriously and appreciate responsible disclosure.

### How to Report

**Please DO NOT report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by:

1. **Email**: Send details to `security@masb-alt.org` (if available)
2. **GitHub Security Advisory**: Use GitHub's private vulnerability reporting feature
3. **Direct Contact**: Contact project maintainers directly through private channels

### What to Include

When reporting a security vulnerability, please include:

- **Description**: Clear description of the vulnerability
- **Impact**: Potential impact and affected components
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Proof of Concept**: If possible, provide a minimal proof of concept
- **Suggested Fix**: If you have ideas for fixing the issue
- **Your Contact Info**: How we can reach you for follow-up

### Response Timeline

We will respond to security reports according to the following timeline:

- **Initial Response**: Within 48 hours of report
- **Preliminary Assessment**: Within 1 week
- **Detailed Investigation**: Within 2 weeks
- **Fix Development**: Timeline depends on complexity
- **Public Disclosure**: After fix is available and deployed

## 🔒 Security Best Practices

### For Users

1. **API Key Management**
   ```bash
   # Use environment variables
   export OPENAI_API_KEY="your-key-here"
   export ANTHROPIC_API_KEY="your-key-here"
   
   # Or use .env file (never commit to git)
   cp .env.example .env
   # Edit .env with your keys
   ```

2. **Database Security**
   ```python
   # Use proper file permissions
   chmod 600 data/masb_alt.db
   
   # Regular backups
   python -m masb_alt.backup --encrypt
   ```

3. **Network Configuration**
   ```python
   # Use secure configurations
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://yourdomain.com"],  # Not "*"
       allow_credentials=True,
       allow_methods=["GET", "POST"],
       allow_headers=["*"],
   )
   ```

### For Developers

1. **Input Validation**
   - Validate all user inputs
   - Sanitize data before database storage
   - Use parameterized queries to prevent SQL injection

2. **Authentication and Authorization**
   - Implement proper authentication for API endpoints
   - Use role-based access control where appropriate
   - Validate user permissions for sensitive operations

3. **Error Handling**
   - Don't expose sensitive information in error messages
   - Log security events appropriately
   - Implement proper exception handling

4. **Dependency Management**
   ```bash
   # Regular security updates
   pip install --upgrade pip
   pip-audit
   safety check
   ```

### For System Administrators

1. **Deployment Security**
   - Use containerization with security scanning
   - Implement network segmentation
   - Regular security assessments and penetration testing
   - Monitor system access and usage patterns

2. **Data Governance**
   - Implement data retention policies
   - Ensure compliance with privacy regulations (GDPR, etc.)
   - Regular security audits
   - Incident response procedures

## 🔍 Security Monitoring

### Automated Checks

The project includes automated security checks:

- **GitHub Security Advisories**: Automatic dependency vulnerability scanning
- **CodeQL Analysis**: Static code analysis for security issues
- **Secret Scanning**: Detection of accidentally committed secrets
- **Dependency Updates**: Automated updates for security patches

### Manual Reviews

Regular manual security reviews include:

- Code reviews focusing on security implications
- Architecture reviews for security design
- Penetration testing of web components
- Security configuration audits

## 📚 Security Resources

### Documentation
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python-security.readthedocs.io/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Streamlit Security](https://docs.streamlit.io/knowledge-base/deploy/authentication-without-sso)

### Tools
- `bandit`: Python security linter
- `safety`: Python dependency vulnerability scanner
- `pip-audit`: Pip package vulnerability scanner
- `semgrep`: Static analysis security tool

## 🏆 Security Hall of Fame

We recognize security researchers and contributors who help improve our security:

<!-- Security contributors will be listed here -->

---

## 📞 Contact

For security-related questions or concerns:

- **Security Email**: security@masb-alt.org (if available)
- **General Issues**: [GitHub Issues](https://github.com/masb-alt/masb-alt/issues)
- **Maintainers**: See [CONTRIBUTING.md](CONTRIBUTING.md) for contact information

Thank you for helping keep MASB-Alt secure! 🔐