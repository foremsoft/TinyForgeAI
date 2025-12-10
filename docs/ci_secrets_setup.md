# CI/CD Secrets Configuration Guide

This guide explains how to configure the GitHub Actions secrets required for the TinyForgeAI CI/CD pipelines.

## Overview

TinyForgeAI uses GitHub Actions for:
- **CI** (`ci.yml`) - Runs tests on every push/PR
- **Docker Build** (`docker_build.yml`) - Builds and pushes Docker images
- **Netlify Deploy** (`deploy_site_netlify.yml`) - Deploys documentation site
- **PyPI Publish** (`publish_pypi.yml`) - Publishes packages to PyPI

## Required Secrets

### 1. Docker (GitHub Container Registry)

**No additional secrets required!** The Docker workflow uses GitHub's built-in `GITHUB_TOKEN` which is automatically provided.

Images are pushed to: `ghcr.io/foremsoft/tinyforgeai`

### 2. Netlify Deployment

For the documentation site deployment, configure these secrets:

| Secret Name | Description | How to Get |
|-------------|-------------|------------|
| `NETLIFY_AUTH_TOKEN` | Personal access token | Netlify Dashboard → User Settings → Applications → Personal Access Tokens |
| `NETLIFY_SITE_ID` | Site API ID | Netlify Dashboard → Site Settings → General → Site Details |

**Setup Steps:**

1. **Create a Netlify Account** (if you don't have one):
   - Go to [netlify.com](https://netlify.com) and sign up

2. **Create a New Site**:
   - Click "Add new site" → "Deploy manually"
   - Or link to your GitHub repo for automatic deployments

3. **Get Site ID**:
   - Go to Site Settings → General → Site Details
   - Copy the "Site ID" (looks like: `12345678-1234-1234-1234-123456789012`)

4. **Generate Auth Token**:
   - Go to User Settings → Applications → Personal Access Tokens
   - Click "New access token"
   - Give it a descriptive name (e.g., "TinyForgeAI Deploy")
   - Copy the generated token

5. **Add Secrets to GitHub**:
   - Go to your repo → Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Add `NETLIFY_AUTH_TOKEN` with your token
   - Add `NETLIFY_SITE_ID` with your site ID

### 3. Test Coverage (Codecov)

For test coverage reporting, configure Codecov:

| Secret Name | Description | How to Get |
|-------------|-------------|------------|
| `CODECOV_TOKEN` | Upload token for Codecov | Codecov Dashboard → Settings → Repository Upload Token |

**Setup Steps:**

1. **Connect Codecov to GitHub**:
   - Go to [codecov.io](https://codecov.io)
   - Sign in with GitHub
   - Add your repository

2. **Get Upload Token**:
   - Go to Settings → General → Repository Upload Token
   - Copy the token

3. **Add Secret to GitHub**:
   - Go to repo Settings → Secrets and variables → Actions
   - Add `CODECOV_TOKEN` with your token

4. **Create codecov.yml** (optional, for custom configuration):
   ```yaml
   coverage:
     precision: 2
     round: down
     status:
       project:
         default:
           target: auto
           threshold: 5%
   ```

### 4. PyPI Publishing

The PyPI workflow uses **Trusted Publishing** (OIDC), which is more secure than API tokens.

**No secrets required!** Instead, configure the PyPI trusted publisher:

**Setup Steps:**

1. **For TestPyPI**:
   - Go to [test.pypi.org](https://test.pypi.org)
   - Log in and go to "Your projects" → "tinyforgeai" → "Settings" → "Publishing"
   - Add a new GitHub publisher:
     - Owner: `foremsoft`
     - Repository: `TinyForgeAI`
     - Workflow name: `publish_pypi.yml`
     - Environment: `testpypi`

2. **For PyPI (Production)**:
   - Go to [pypi.org](https://pypi.org)
   - Same steps as above, but set:
     - Environment: `pypi`

3. **Create GitHub Environments**:
   - Go to repo Settings → Environments
   - Create environment named `testpypi`
   - Create environment named `pypi`
   - Optionally add required reviewers for the `pypi` environment

## GitHub Environments Setup

The PyPI workflow uses GitHub Environments for deployment protection:

### Creating Environments

1. Go to **Settings → Environments**
2. Click **New environment**
3. Create two environments:

**testpypi**:
- Name: `testpypi`
- Protection rules: None (for testing)

**pypi**:
- Name: `pypi`
- Protection rules (recommended):
  - Required reviewers: Add maintainers who must approve production releases
  - Wait timer: Optional delay before deployment

## Verification

After configuring secrets, verify your setup:

### Test Docker Build

```bash
# Trigger manually
gh workflow run "Build and Push Docker Image" --ref main
```

### Test Netlify Deploy

```bash
# Push to main branch, or trigger manually
gh workflow run "Deploy to Netlify" --ref main
```

### Test PyPI Publish

```bash
# Test with TestPyPI first
gh workflow run "Publish to PyPI" -f publish_to=testpypi
```

## Troubleshooting

### Docker: "Permission denied" or "unauthorized"

- Ensure workflow has `packages: write` permission
- Check that `GITHUB_TOKEN` is being used correctly

### Netlify: "Invalid token" or "Site not found"

- Verify `NETLIFY_AUTH_TOKEN` hasn't expired
- Check `NETLIFY_SITE_ID` matches exactly (no extra spaces)
- Ensure the token has deploy permissions

### PyPI: "Trusted publishing not configured"

- Verify the publisher configuration on PyPI exactly matches:
  - Owner name (case-sensitive)
  - Repository name (case-sensitive)
  - Workflow filename (exact match)
  - Environment name (exact match)

### PyPI: "Environment not found"

- Create the `testpypi` and `pypi` environments in GitHub repo settings
- Environments are case-sensitive

## Security Best Practices

1. **Never commit secrets** to the repository
2. **Use environment protection** for production deployments
3. **Rotate tokens periodically** (every 90 days recommended)
4. **Limit token scopes** to minimum required permissions
5. **Review workflow runs** for unexpected behavior
6. **Enable branch protection** on main branch

## Quick Reference

| Workflow | Secrets Needed | Notes |
|----------|----------------|-------|
| CI | `CODECOV_TOKEN` (optional) | Coverage badge won't work without it |
| Docker Build | None | Uses `GITHUB_TOKEN` for GHCR |
| Netlify Deploy | `NETLIFY_AUTH_TOKEN`, `NETLIFY_SITE_ID` | Manual setup required |
| PyPI Publish | None | Uses OIDC Trusted Publishing |

## Links

- [GitHub Actions Secrets Documentation](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [Netlify CLI Authentication](https://docs.netlify.com/cli/get-started/#authentication)
- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [GitHub Environments](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment)
