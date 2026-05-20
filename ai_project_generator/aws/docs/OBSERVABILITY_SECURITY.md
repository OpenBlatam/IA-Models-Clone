# Observability & Security

Complete guide for advanced observability with Prometheus/Grafana and security hardening.

## 📋 Table of Contents

1. [Prometheus & Grafana](#prometheus--grafana)
2. [Security Hardening](#security-hardening)
3. [API Gateway](#api-gateway)
4. [Chaos Engineering](#chaos-engineering)

## 📊 Prometheus & Grafana

### Overview

Advanced observability stack with Prometheus for metrics collection and Grafana for visualization.

### Installation

```bash
# Install Prometheus and Grafana
sudo /opt/ai-project-generator/scripts/prometheus_setup.sh install

# Start services
sudo /opt/ai-project-generator/scripts/prometheus_setup.sh start

# Check status
sudo /opt/ai-project-generator/scripts/prometheus_setup.sh status
```

### Access

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Features

- ✅ **Metrics Collection**: Prometheus scrapes metrics from multiple sources
- ✅ **Visualization**: Grafana dashboards for monitoring
- ✅ **Exporters**: Node Exporter, Nginx, Redis, Docker exporters
- ✅ **Alerting**: Prometheus alerting rules
- ✅ **Long-term Storage**: Configurable retention policies

### Metrics Collected

- **System Metrics**: CPU, memory, disk, network
- **Application Metrics**: Request rate, response time, error rate
- **Nginx Metrics**: Request/response stats, upstream health
- **Redis Metrics**: Memory usage, commands, connections
- **Docker Metrics**: Container stats, resource usage

### Grafana Dashboards

Import these dashboards from grafana.com:
- Node Exporter Full (ID: 1860)
- Docker Container & Host Metrics (ID: 11074)
- Redis Dashboard (ID: 12708)
- Nginx Metrics (ID: 12708)

### Configuration

Prometheus configuration: `/opt/prometheus/prometheus.yml`

```yaml
scrape_configs:
  - job_name: 'ai-project-generator'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['localhost:8020']
```

## 🔒 Security Hardening

### Overview

Comprehensive security hardening following industry best practices.

### Usage

```bash
# Full security hardening
sudo /opt/ai-project-generator/scripts/security_hardening.sh all

# Individual components
sudo /opt/ai-project-generator/scripts/security_hardening.sh firewall
sudo /opt/ai-project-generator/scripts/security_hardening.sh fail2ban
sudo /opt/ai-project-generator/scripts/security_hardening.sh ssh

# Run security scan
sudo /opt/ai-project-generator/scripts/security_hardening.sh scan
```

### Features

- ✅ **Firewall (UFW)**: Restrictive firewall rules
- ✅ **Fail2ban**: Intrusion prevention
- ✅ **Automatic Updates**: Security updates automation
- ✅ **SSH Hardening**: Secure SSH configuration
- ✅ **Audit Logging**: System audit with auditd
- ✅ **AppArmor**: Application security profiles
- ✅ **Security Tools**: rkhunter, chkrootkit, ClamAV

### Security Measures

#### Firewall Configuration

- Default deny incoming
- Allow only necessary ports (22, 80, 443, 8020)
- Rate limiting on connections

#### Fail2ban

- SSH brute force protection
- Nginx rate limiting
- Configurable ban duration

#### SSH Hardening

- Disable root login
- Disable password authentication
- Key-based authentication only
- Connection timeouts
- Max authentication attempts: 3

#### Automatic Security Updates

- Daily package list updates
- Automatic security patches
- Unattended upgrades

#### Security Scanning

Tools included:
- **rkhunter**: Rootkit detection
- **chkrootkit**: Rootkit scanner
- **Lynis**: Security auditing
- **ClamAV**: Antivirus scanning

## 🚪 API Gateway

### Overview

API Gateway with rate limiting, versioning, and monitoring.

### Installation

```bash
# Setup API Gateway
sudo /opt/ai-project-generator/scripts/api_gateway_setup.sh setup

# Reload configuration
sudo /opt/ai-project-generator/scripts/api_gateway_setup.sh reload

# Monitor API metrics
sudo /opt/ai-project-generator/scripts/api_gateway_setup.sh monitor
```

### Features

- ✅ **Rate Limiting**: Per-IP rate limiting
- ✅ **API Versioning**: Support for multiple API versions
- ✅ **Security Headers**: Security headers injection
- ✅ **Connection Limiting**: Max connections per IP
- ✅ **Health Checks**: Dedicated health check endpoint
- ✅ **Metrics Endpoint**: Internal metrics access
- ✅ **Error Handling**: Custom error pages

### Rate Limits

- **API Endpoints**: 10 requests/second (burst: 20)
- **Auth Endpoints**: 5 requests/second (burst: 5)
- **Connection Limit**: 10 connections per IP

### API Versioning

- `/api/v1/` - Version 1 endpoints
- `/api/v2/` - Version 2 endpoints (future)

### Security Headers

- `X-Frame-Options: SAMEORIGIN`
- `X-Content-Type-Options: nosniff`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000`

### Monitoring

Monitor API metrics:
```bash
/opt/ai-project-generator/scripts/monitor_api.sh
```

Metrics tracked:
- Rate limit hits
- Total requests
- Error rate
- Error percentage

## 🔥 Chaos Engineering

### Overview

Test system resilience through controlled failure injection.

### Installation

```bash
# Install chaos tools
sudo /opt/ai-project-generator/scripts/chaos_engineering.sh install
```

### Usage

```bash
# CPU chaos (60s, 50% load)
sudo /opt/ai-project-generator/scripts/chaos_engineering.sh cpu 60 50

# Memory chaos (60s, 512MB)
sudo /opt/ai-project-generator/scripts/chaos_engineering.sh memory 60 512

# Network chaos (60s, 100ms latency, 5% packet loss)
sudo /opt/ai-project-generator/scripts/chaos_engineering.sh network 60 100 5

# Disk I/O chaos (60s)
sudo /opt/ai-project-generator/scripts/chaos_engineering.sh disk 60

# Service chaos (stop service for 30s)
sudo /opt/ai-project-generator/scripts/chaos_engineering.sh service ai-project-generator 30

# Container chaos (stop container for 30s)
sudo /opt/ai-project-generator/scripts/chaos_engineering.sh container ai-project-generator 30

# Run full chaos test suite
sudo /opt/ai-project-generator/scripts/chaos_engineering.sh suite
```

### Chaos Types

#### CPU Chaos
- Injects CPU load
- Tests application under CPU stress
- Monitors response times

#### Memory Chaos
- Injects memory pressure
- Tests memory handling
- Monitors memory usage

#### Network Chaos
- Adds network latency
- Simulates packet loss
- Tests network resilience

#### Disk I/O Chaos
- Creates disk I/O pressure
- Tests disk performance
- Monitors I/O metrics

#### Service Chaos
- Stops application service
- Tests auto-recovery
- Validates health checks

#### Container Chaos
- Stops Docker containers
- Tests container orchestration
- Validates restart policies

### Monitoring During Chaos

The script automatically monitors:
- Application health
- Success/failure rates
- Recovery time
- Resilience metrics

### Best Practices

1. **Start Small**: Begin with short durations and low intensity
2. **Monitor Closely**: Watch metrics during chaos
3. **Test Regularly**: Run chaos tests weekly/monthly
4. **Document Results**: Keep records of chaos test results
5. **Improve Based on Results**: Fix issues found during chaos tests

## 🔗 Integration

### Prometheus Integration

- Scrapes application metrics
- Integrates with CloudWatch
- Supports custom metrics
- Long-term storage

### Security Integration

- Integrates with CloudWatch Logs
- Sends alerts via SNS
- Compliance reporting
- Audit trail

### API Gateway Integration

- Works with ALB
- Integrates with WAF
- Supports SSL/TLS
- Rate limiting logs

## 📊 Monitoring Dashboard

Create dashboards for:
- System metrics (CPU, memory, disk)
- Application metrics (requests, errors)
- Security events (fail2ban, audit logs)
- API metrics (rate limits, errors)
- Chaos test results

## 🔔 Alerting

Configure alerts for:
- High error rates
- Rate limit violations
- Security events
- System resource exhaustion
- Chaos test failures

## 📈 Best Practices

### Observability

1. Collect metrics from all services
2. Use structured logging
3. Set up dashboards
4. Configure alerts
5. Review metrics regularly

### Security

1. Keep systems updated
2. Use strong authentication
3. Monitor security logs
4. Run regular scans
5. Follow least privilege

### API Gateway

1. Set appropriate rate limits
2. Monitor API usage
3. Version APIs properly
4. Use security headers
5. Log all requests

### Chaos Engineering

1. Test in staging first
2. Start with non-critical services
3. Have rollback plans
4. Document all tests
5. Learn from failures

---

**Version**: 1.0  
**Status**: Production Ready ✅  
**Last Updated**: 2024



