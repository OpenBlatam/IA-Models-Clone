# Installation and Setup Guide

This guide will help you set up the 3D Prototype AI deployment environment.

## 🚀 Quick Setup

### Automatic Setup (Recommended)

```bash
cd cloud
./scripts/setup.sh
```

Or using Makefile:

```bash
cd cloud
make setup
```

This will:
- Make all scripts executable
- Create necessary directories
- Set up configuration files
- Check dependencies
- Verify script syntax

## 📋 Prerequisites

### Required

- **Bash** 4.0+ (usually pre-installed on Linux/macOS)
- **Git** (for cloning the repository)
- **curl** (for health checks and downloads)

### Optional (but recommended)

- **AWS CLI** - For AWS deployments
- **Terraform** - For Infrastructure as Code
- **Ansible** - For configuration management
- **Docker** - For containerized deployment
- **shellcheck** - For script linting

## 🔧 Manual Setup

### 1. Fix Script Permissions

```bash
cd cloud
./scripts/fix_permissions.sh
```

Or:

```bash
make fix-perms
```

### 2. Check Environment

```bash
./scripts/check_environment.sh
```

Or:

```bash
make check-env
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your settings
vim .env  # or use your preferred editor
```

### 4. Configure Terraform (if using)

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
vim terraform.tfvars
```

### 5. Configure Ansible (if using)

```bash
cd ansible/inventory
cp ec2.ini.example ec2.ini
vim ec2.ini
```

## 🐧 Platform-Specific Instructions

### Linux

Most scripts should work out of the box:

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y bash curl git

# Optional: Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Run setup
cd cloud
./scripts/setup.sh
```

### macOS

```bash
# Install dependencies using Homebrew
brew install bash curl git

# Optional: Install AWS CLI
brew install awscli

# Optional: Install Terraform
brew install terraform

# Optional: Install Ansible
brew install ansible

# Run setup
cd cloud
./scripts/setup.sh
```

### Windows

#### Option 1: Git Bash (Recommended)

1. Install [Git for Windows](https://git-scm.com/download/win)
2. Open Git Bash
3. Run setup:

```bash
cd cloud
./scripts/setup.sh
```

#### Option 2: WSL (Windows Subsystem for Linux)

1. Install WSL:

```powershell
wsl --install
```

2. Open WSL and follow Linux instructions

#### Option 3: PowerShell

Some scripts may work in PowerShell, but bash is recommended.

## ✅ Verification

After setup, verify everything works:

```bash
# Check environment
./scripts/check_environment.sh

# Validate configuration
./scripts/validate.sh

# Test scripts
./scripts/test_scripts.sh
```

## 🔍 Troubleshooting

### Scripts are not executable

```bash
# Fix permissions
./scripts/fix_permissions.sh

# Or manually
chmod +x scripts/*.sh
chmod +x scripts/lib/*.sh
```

### "Permission denied" errors

On Linux/macOS, you may need to make scripts executable:

```bash
chmod +x scripts/*.sh
```

On Windows with Git Bash, scripts should work after running `setup.sh`.

### Bash not found

**Linux:**
```bash
sudo apt-get install bash
```

**macOS:**
```bash
brew install bash
```

**Windows:**
Install Git for Windows or use WSL.

### AWS CLI not configured

```bash
aws configure
```

Enter your:
- AWS Access Key ID
- AWS Secret Access Key
- Default region
- Default output format

### Script syntax errors

Run validation:

```bash
./scripts/validate.sh
```

Or manually check:

```bash
bash -n scripts/deploy.sh
```

## 📚 Next Steps

After setup:

1. **Configure AWS credentials** (if deploying to AWS)
2. **Edit .env file** with your settings
3. **Configure Terraform/Ansible** (if using)
4. **Validate configuration**: `./scripts/validate.sh`
5. **Deploy**: `./scripts/deploy.sh`

## 🆘 Getting Help

If you encounter issues:

1. Run environment check: `./scripts/check_environment.sh`
2. Check logs in `logs/` directory
3. Review [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
4. Check script syntax: `bash -n scripts/<script>.sh`

---

**Last Updated**: 2024-01-XX

