# Deployment Role

Ansible role for deploying Music Analyzer AI application.

## Requirements

- Docker installed (use docker role)
- Application files available
- Docker images available

## Role Variables

```yaml
app_name: "music-analyzer-ai"
app_directory: "/opt/music-analyzer-ai"
deployment_user: "{{ ansible_user }}"
docker_compose_file: "docker-compose.yml"
env_file: ".env"
deployment_scripts:
  - deploy.sh
  - health-check.sh
docker_images:
  - backend-image.tar.gz
  - frontend-image.tar.gz
backend_port: 8000
docker_pull: false
docker_recreate: "always"
```

## Dependencies

- docker role

## Example Playbook

```yaml
- hosts: all
  roles:
    - role: docker
    - role: deployment
      vars:
        app_directory: "/opt/music-analyzer-ai"
        docker_compose_file: "docker-compose.yml"
```

## Tags

- `deploy` - Deploy application
- `application` - Application tasks

## Handlers

- `restart application` - Restart application containers

## License

MIT




