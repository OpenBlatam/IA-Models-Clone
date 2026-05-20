# Nginx Infrastructure

<div align="center">

![Status](https://img.shields.io/badge/status-production--ready-success.svg)
![Role](https://img.shields.io/badge/role-reverse--proxy-blue.svg)
![Security](https://img.shields.io/badge/security-SSL--TLS-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**High-performance Nginx configuration for reverse proxying, load balancing, and production-grade security.**

[Overview](#-overview) •
[Features](#-key-features) •
[Configuration](#-configuration) •
[Integration](#-integration) •
[Contributing](#-contributing)

</div>

---

## 📋 Overview

The **Nginx Infrastructure** module provides the primary edge layer for the Blatam Academy ecosystem. It is responsible for routing incoming traffic to appropriate microservices, managing SSL termination, providing high-efficiency caching, and balancing load between redundant service instances.

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **Reverse Proxy** | Intelligent request routing to backend services. |
| **Load Balancing** | Traffic distribution across multiple service containers. |
| **SSL/TLS Hardening** | Secure communication with modern cipher suites. |
| **Static Caching** | High-speed delivery of static assets and API responses. |

## 📁 Structure

```
nginx/
└── nginx.conf              # Master Nginx configuration for production and discovery
```

## 🔧 Configuration

All configurations reside in `nginx.conf`. In a production environment, this is typically orchestrated via **Docker Compose**, which mounts the configuration file into the standard Nginx container.

## 🔗 Integration

This infrastructure component coordinates with:
- **Onyx Discovery Layer**: To map service names to internal IP addresses.
- **Docker Compose**: For container registration and lifespan management.
- **All Backend Services**: Acts as the single point of entry for the API and Frontend.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](../../../CONTRIBUTING.md) for details.

---

<div align="center">
  <b>Built with ❤️ by Blatam Academy</b><br>
  Part of the Onyx Server Architecture<br>
  <a href="../README.md">← Back to Main README</a>
</div>
