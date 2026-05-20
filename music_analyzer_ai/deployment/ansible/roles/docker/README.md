# Docker Role

Ansible role for installing and configuring Docker on target hosts.

## Requirements

- Ansible 2.9+
- Target OS: Ubuntu 20.04+, Debian 11+, Amazon Linux 2, RHEL 8+

## Role Variables

```yaml
docker_version: "latest"  # Docker version to install
docker_compose_version: "2.23.0"  # Docker Compose version
docker_user: "{{ ansible_user }}"  # User to add to docker group
```

## Dependencies

None

## Example Playbook

```yaml
- hosts: all
  roles:
    - role: docker
      vars:
        docker_version: "24.0"
```

## Tags

- `docker` - Install Docker
- `setup` - Setup tasks

## Handlers

- `restart docker` - Restart Docker service

## License

MIT




