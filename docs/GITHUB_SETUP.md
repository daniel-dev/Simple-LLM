# 🐙 GitHub Installation Guide for LLM Factory

This guide provides step-by-step instructions for GitHub users to set up and run LLM Factory.

## 🚀 Quick Setup for GitHub

### Prerequisites Check

Before starting, ensure you have:

- [ ] **Git** installed and configured
- [ ] **Python 3.13+** installed
- [ ] **GitHub account** with repository access
- [ ] **4GB+ RAM** available
- [ ] **2GB+ disk space** for models and data

### 📋 Installation Steps

#### Step 1: Clone from GitHub

```bash
# Option A: HTTPS (recommended for most users)
git clone https://github.com/YOUR_USERNAME/llm-factory.git
cd llm-factory

# Option B: SSH (if you have SSH keys configured)
git clone git@github.com:YOUR_USERNAME/llm-factory.git
cd llm-factory

# Option C: GitHub token (for private repos or automation)
git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/llm-factory.git
cd llm-factory
```

#### Step 2: Environment Setup

```bash
# Check Python version
python --version  # Should be 3.13+

# Create virtual environment
python -m venv llm-factory-env

# Activate virtual environment
# Windows:
llm-factory-env\Scripts\activate
# macOS/Linux:
source llm-factory-env/bin/activate

# Verify activation (should show virtual env path)
python -c "import sys; print(sys.prefix)"
```

#### Step 3: GitHub Actions Configuration

```bash
# Create GitHub Actions workflow directory
mkdir -p .github/workflows

# Copy the example GitHub Actions workflow
cp docs/github-actions-example.yml .github/workflows/ci.yml

# Customize for your project
# Edit .github/workflows/ci.yml to match your GitHub project settings
```

> **🚀 GitHub Actions Features:**
> - Automated testing on every commit and pull request
> - Docker image building and GitHub Container Registry push
> - GitHub Pages deployment for documentation
> - Security scanning (CodeQL, Dependabot)
> - Workflow caching for faster builds

#### Step 4: Install Dependencies

```bash
# Upgrade pip first (recommended)
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import torch, transformers, flask, matplotlib; print('✅ All packages installed!')"
```

#### Step 5: Run the Pipeline

```bash
# Execute the enhanced pipeline
python run_simple_finetuning.py

# This will generate detailed analytics and model data
# Output will be saved to data/processed/ and models/trained/
```

#### Step 6: Start the Web Application

```bash
# Launch the Flask web interface
python app.py

# Access the dashboard at: http://127.0.0.1:5000
```

### 🔧 GitHub-Specific Configuration

#### Personal Access Tokens

For private repositories or GitHub Actions:

1. Go to **Settings** → **Developer settings** → **Personal access tokens**
2. Generate new token with scopes: `repo`, `read:packages`, `write:packages`
3. Use token for clone: `git clone https://TOKEN@github.com/user/repo.git`

#### SSH Key Setup

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key to GitHub
cat ~/.ssh/id_ed25519.pub
# Paste into GitHub Settings → SSH and GPG keys
```

#### GitHub CLI Setup

```bash
# Install GitHub CLI
# Windows: winget install GitHub.cli
# macOS: brew install gh
# Linux: See https://cli.github.com/

# Authenticate
gh auth login

# Clone repository
gh repo clone username/llm-factory
```

### 🐳 GitHub Container Registry

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Build and push Docker image
docker build -t ghcr.io/username/llm-factory:latest .
docker push ghcr.io/username/llm-factory:latest
```

### 📦 GitHub Packages (PyPI)

```bash
# Setup .pypirc for GitHub Packages
echo "[distutils]
index-servers = github

[github]
repository = https://upload.pypi.org/legacy/
username = __token__
password = YOUR_GITHUB_TOKEN" > ~/.pypirc

# Build and upload package
python setup.py sdist bdist_wheel
twine upload --repository github dist/*
```

### 🚀 GitHub Actions Advanced Configuration

#### Matrix Strategy

```yaml
strategy:
  matrix:
    python-version: ['3.11', '3.12', '3.13']
    os: [ubuntu-latest, windows-latest, macos-latest]
```

#### Conditional Workflows

```yaml
if: github.event_name == 'push' && github.ref == 'refs/heads/main'
```

#### Secrets Management

- Store sensitive data in **Settings** → **Secrets and variables** → **Actions**
- Access in workflows: `${{ secrets.SECRET_NAME }}`

### 🛡️ Security Best Practices

#### Dependabot Configuration

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
```

#### CodeQL Analysis

Automatically enabled for public repositories, or add to workflow:

```yaml
- uses: github/codeql-action/init@v2
  with:
    languages: python
```

### 📊 GitHub Pages Deployment

```bash
# Enable GitHub Pages in repository settings
# Choose source: GitHub Actions

# Your documentation will be available at:
# https://username.github.io/llm-factory
```

### 🔍 Troubleshooting GitHub Issues

#### Common Authentication Problems

1. **Two-Factor Authentication**: Use personal access tokens instead of passwords
2. **SSH Connection Issues**: Test with `ssh -T git@github.com`
3. **Permission Denied**: Check repository access and SSH key configuration

#### GitHub Actions Failures

1. **Secrets Not Available**: Check if secrets are properly configured
2. **Workflow Permissions**: Ensure GITHUB_TOKEN has necessary permissions
3. **Runner Capacity**: Public repositories have usage limits

#### Large File Issues

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.bin" "*.safetensors" "*.pt"

# Add .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

### 🎯 Next Steps

1. **Customize Workflows**: Modify `.github/workflows/ci.yml` for your needs
2. **Setup Environments**: Configure staging/production environments
3. **Add Tests**: Create comprehensive test suite
4. **Documentation**: Use GitHub Wiki or Pages for detailed docs
5. **Community**: Setup issue templates and contributing guidelines

### 📚 Additional Resources

- **GitHub Docs**: [docs.github.com](https://docs.github.com)
- **GitHub Actions**: [docs.github.com/actions](https://docs.github.com/actions)
- **GitHub CLI**: [cli.github.com](https://cli.github.com)
- **Git LFS**: [git-lfs.github.io](https://git-lfs.github.io)

---

**🏭 LLM Factory** is now fully configured for GitHub! 🐙✨
