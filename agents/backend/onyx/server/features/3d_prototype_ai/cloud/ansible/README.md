# Ansible Deployment Guide

This directory contains Ansible playbooks and roles for deploying and configuring the 3D Prototype AI application on EC2 instances.

## 📁 Structure

```
ansible/
├── group_vars/           # Environment-specific variables
│   ├── all.yml          # Common variables for all environments
│   ├── production.yml   # Production environment variables
│   ├── staging.yml      # Staging environment variables
│   └── development.yml  # Development environment variables
├── host_vars/           # Host-specific variables (optional)
├── inventory/           # Inventory files
│   └── ec2.ini.example  # Example inventory
├── playbooks/           # Ansible playbooks
│   ├── deploy.yml       # Main deployment playbook
│   └── templates/       # Jinja2 templates
└── roles/               # Ansible roles
    ├── docker/          # Docker installation and configuration
    ├── nginx/           # Nginx configuration
    ├── monitoring/      # Monitoring and logging setup
    └── security/        # Security hardening
```

## 🚀 Quick Start

### 1. Configure Inventory

Copy and edit the inventory file:

```bash
cp inventory/ec2.ini.example inventory/ec2.ini
# Edit inventory/ec2.ini with your instance details
```

### 2. Set Environment Variables

Edit `group_vars/all.yml` or environment-specific files:

```bash
# For production
vim group_vars/production.yml

# For staging
vim group_vars/staging.yml
```

### 3. Run Deployment

```bash
# Deploy to all hosts
ansible-playbook -i inventory/ec2.ini playbooks/deploy.yml

# Deploy to specific environment
ansible-playbook -i inventory/ec2.ini playbooks/deploy.yml -e "environment=production"

# Deploy with specific tags
ansible-playbook -i inventory/ec2.ini playbooks/deploy.yml --tags "docker,nginx"
```

## 🎯 Roles

### Security Role

Hardens the system with:
- Automatic security updates
- UFW firewall configuration
- Fail2Ban setup
- SSH hardening
- Timezone and locale configuration

**Usage:**
```bash
ansible-playbook -i inventory/ec2.ini playbooks/deploy.yml --tags security
```

### Docker Role

Installs and configures Docker:
- Docker CE installation
- Docker Compose setup
- User group configuration

**Usage:**
```bash
ansible-playbook -i inventory/ec2.ini playbooks/deploy.yml --tags docker
```

### Nginx Role

Configures Nginx reverse proxy:
- Site configuration
- Security headers
- Logging setup
- SSL ready (when certificates are available)

**Usage:**
```bash
ansible-playbook -i inventory/ec2.ini playbooks/deploy.yml --tags nginx
```

### Monitoring Role

Sets up monitoring and logging:
- CloudWatch agent
- Log rotation
- Monitoring tools

**Usage:**
```bash
ansible-playbook -i inventory/ec2.ini playbooks/deploy.yml --tags monitoring
```

## 📋 Variables

### Common Variables (all.yml)

- `app_user`: Application user (default: ubuntu)
- `app_dir`: Application directory (default: /opt/3d-prototype-ai)
- `app_port`: Application port (default: 8030)
- `use_docker`: Use Docker deployment (default: true)
- `cloudwatch_enabled`: Enable CloudWatch (default: true)

### Environment-Specific Variables

Each environment (production, staging, development) has its own variables file that overrides common settings.

## 🔒 Security

### Using Ansible Vault

For sensitive data, use Ansible Vault:

```bash
# Create encrypted variable file
ansible-vault create group_vars/production/vault.yml

# Edit encrypted file
ansible-vault edit group_vars/production/vault.yml

# Use in playbook
ansible-playbook -i inventory/ec2.ini playbooks/deploy.yml --ask-vault-pass
```

## 🏷️ Tags

Use tags to run specific parts of the playbook:

```bash
# Only security hardening
ansible-playbook -i inventory/ec2.ini playbooks/deploy.yml --tags security

# Only infrastructure (docker, nginx)
ansible-playbook -i inventory/ec2.ini playbooks/deploy.yml --tags infrastructure

# Skip security
ansible-playbook -i inventory/ec2.ini playbooks/deploy.yml --skip-tags security
```

Available tags:
- `security`: Security hardening tasks
- `docker`: Docker installation and configuration
- `nginx`: Nginx configuration
- `monitoring`: Monitoring setup
- `infrastructure`: All infrastructure tasks

## ✅ Validation

Validate playbooks before running:

```bash
# Syntax check
ansible-playbook --syntax-check -i inventory/ec2.ini playbooks/deploy.yml

# Lint check
ansible-lint playbooks/deploy.yml

# Dry run (check mode)
ansible-playbook --check -i inventory/ec2.ini playbooks/deploy.yml
```

## 📊 Best Practices

1. **Use group_vars**: Environment-specific configuration
2. **Use roles**: Modular and reusable configurations
3. **Use tags**: Flexible task execution
4. **Use vault**: Secure sensitive data
5. **Validate**: Always validate before running
6. **Idempotent**: All tasks should be idempotent
7. **Documentation**: Document custom variables and roles

## 🔧 Troubleshooting

### Common Issues

1. **SSH Connection Issues**
   - Verify SSH key permissions
   - Check security group rules
   - Verify instance is running

2. **Permission Errors**
   - Ensure `become: yes` is set
   - Check sudo permissions

3. **Variable Not Found**
   - Check group_vars files
   - Verify variable names
   - Check variable precedence

### Debug Mode

Run with verbose output:

```bash
ansible-playbook -i inventory/ec2.ini playbooks/deploy.yml -vvv
```

## 📚 Additional Resources

- [Ansible Documentation](https://docs.ansible.com/)
- [Ansible Best Practices](https://docs.ansible.com/ansible/latest/user_guide/playbooks_best_practices.html)
- [Ansible Vault](https://docs.ansible.com/ansible/latest/user_guide/vault.html)

