# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

---

## 🔒 Security Practices in MoodMentor

1. **Password Storage:**
   - Passwords are never stored in plain text.
   - Credentials use random per-user cryptographic salts and PBKDF2-HMAC-SHA256 key derivation with 120,000 iterations.
2. **SQL Injection Defense:**
   - All SQLite queries use parameterized placeholders (`?`), preventing SQL injection vulnerabilities.
3. **Input Sanitization:**
   - Ingested texts and file uploads are strictly validated and checked against UTF-8 encodings and header integrity.
4. **Environment Isolation:**
   - Secrets and local databases (`wellness.db`) are excluded from version control via `.gitignore`.

---

## 🛡️ Reporting a Vulnerability

If you discover a security vulnerability within MoodMentor, please report it privately:
- Contact the maintainer: **Deeksha Ganesh** ([@DeekshaG96](https://github.com/DeekshaG96))
- Email: `dg575265@gmail.com`

Please do not open public issues for sensitive security vulnerabilities.
