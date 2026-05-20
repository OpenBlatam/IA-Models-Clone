"""
PDF Variantes Deployment Configuration
Configuración de despliegue para el sistema PDF Variantes
"""

# Configuración de despliegue
deployment_config = {
    "environment": "production",
    "docker": {
        "enabled": True,
        "compose_file": "docker-compose.yml",
        "image_name": "pdf-variantes",
        "tag": "latest",
        "registry": "your-registry.com",
        "build_args": {
            "PYTHON_VERSION": "3.11",
            "APP_VERSION": "2.0.0"
        }
    },
    "kubernetes": {
        "enabled": False,
        "namespace": "pdf-variantes",
        "config_file": "k8s-config.yaml",
        "replicas": 3,
        "resources": {
            "requests": {
                "memory": "2Gi",
                "cpu": "1000m",
                "nvidia.com/gpu": 1
            },
            "limits": {
                "memory": "4Gi",
                "cpu": "2000m",
                "nvidia.com/gpu": 1
            }
        }
    },
    "database": {
        "type": "postgresql",
        "host": "localhost",
        "port": 5432,
        "name": "pdf_variantes",
        "user": "pdf_user",
        "password": "pdf_password",
        "pool_size": 10,
        "max_overflow": 20,
        "ssl_mode": "prefer"
    },
    "redis": {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": None,
        "max_connections": 100,
        "socket_timeout": 5
    },
    "monitoring": {
        "enabled": True,
        "grafana": {
            "enabled": True,
            "port": 3000,
            "admin_user": "admin",
            "admin_password": "admin"
        },
        "prometheus": {
            "enabled": True,
            "port": 9090,
            "retention": "30d"
        },
        "jaeger": {
            "enabled": False,
            "port": 14268
        }
    },
    "ssl": {
        "enabled": False,
        "cert_path": "ssl/cert.pem",
        "key_path": "ssl/key.pem",
        "ca_path": "ssl/ca.pem"
    },
    "load_balancer": {
        "enabled": True,
        "type": "nginx",
        "port": 80,
        "ssl_port": 443,
        "upstream_servers": [
            "localhost:8000",
            "localhost:8001",
            "localhost:8002"
        ]
    },
    "backup": {
        "enabled": True,
        "schedule": "0 2 * * *",  # Daily at 2 AM
        "retention_days": 30,
        "storage": {
            "type": "s3",
            "bucket": "pdf-variantes-backups",
            "region": "us-east-1"
        }
    },
    "security": {
        "rate_limiting": {
            "enabled": True,
            "requests_per_minute": 60,
            "requests_per_hour": 1000
        },
        "cors": {
            "enabled": True,
            "origins": ["https://yourdomain.com"],
            "methods": ["GET", "POST", "PUT", "DELETE"],
            "headers": ["*"]
        },
        "authentication": {
            "enabled": True,
            "jwt_secret": "your-jwt-secret",
            "token_expiry": 3600,
            "refresh_token_expiry": 86400
        }
    },
    "performance": {
        "cache": {
            "enabled": True,
            "ttl": 3600,
            "max_size": "1GB",
            "backend": "redis"
        },
        "compression": {
            "enabled": True,
            "level": 6,
            "min_size": 1024
        },
        "gzip": {
            "enabled": True,
            "level": 6
        }
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "logs/pdf_variantes.log",
        "max_size": "100MB",
        "backup_count": 5,
        "rotation": "daily"
    },
    "ai": {
        "openai": {
            "api_key": "your-openai-api-key",
            "model": "gpt-4",
            "max_tokens": 4000,
            "temperature": 0.7
        },
        "anthropic": {
            "api_key": "your-anthropic-api-key",
            "model": "claude-3-opus",
            "max_tokens": 4000,
            "temperature": 0.7
        },
        "huggingface": {
            "api_key": "your-huggingface-api-key",
            "model": "distilbert-base-uncased",
            "max_length": 512
        }
    },
    "blockchain": {
        "enabled": False,
        "network": "ethereum",
        "rpc_url": "https://goerli.infura.io/v3/YOUR_PROJECT_ID",
        "private_key": "your-private-key",
        "contract_address": "0x...",
        "gas_limit": 1000000,
        "gas_price": "20"
    },
    "plugins": {
        "enabled": True,
        "directory": "plugins",
        "auto_load": True,
        "hot_reload": True,
        "sandbox": True
    },
    "quantum": {
        "enabled": False,
        "backend": "qasm_simulator",
        "shots": 1024,
        "optimization_level": 3
    },
    "web3": {
        "enabled": False,
        "ipfs_gateway": "https://ipfs.io/ipfs/",
        "nft_contract": "0x...",
        "marketplace_contract": "0x..."
    },
    "scaling": {
        "horizontal": {
            "enabled": True,
            "min_replicas": 3,
            "max_replicas": 10,
            "cpu_threshold": 70,
            "memory_threshold": 80
        },
        "vertical": {
            "enabled": False,
            "cpu_limit": "2000m",
            "memory_limit": "4Gi"
        }
    },
    "health_checks": {
        "enabled": True,
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "endpoints": [
            "/health",
            "/metrics",
            "/api/v1/health/detailed"
        ]
    },
    "alerts": {
        "enabled": True,
        "channels": ["email", "slack", "webhook"],
        "thresholds": {
            "cpu_usage": 80,
            "memory_usage": 85,
            "disk_usage": 90,
            "error_rate": 5,
            "response_time": 1000
        }
    }
}

# Configuración por entorno
environments = {
    "development": {
        "environment": "development",
        "debug": True,
        "log_level": "DEBUG",
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "pdf_variantes_dev"
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 1
        },
        "monitoring": {
            "enabled": False
        },
        "ssl": {
            "enabled": False
        },
        "scaling": {
            "horizontal": {
                "enabled": False
            }
        }
    },
    "staging": {
        "environment": "staging",
        "debug": False,
        "log_level": "INFO",
        "database": {
            "host": "staging-db.example.com",
            "port": 5432,
            "name": "pdf_variantes_staging"
        },
        "redis": {
            "host": "staging-redis.example.com",
            "port": 6379,
            "db": 0
        },
        "monitoring": {
            "enabled": True
        },
        "ssl": {
            "enabled": True
        },
        "scaling": {
            "horizontal": {
                "enabled": True,
                "min_replicas": 2,
                "max_replicas": 5
            }
        }
    },
    "production": {
        "environment": "production",
        "debug": False,
        "log_level": "INFO",
        "database": {
            "host": "prod-db.example.com",
            "port": 5432,
            "name": "pdf_variantes_prod"
        },
        "redis": {
            "host": "prod-redis.example.com",
            "port": 6379,
            "db": 0
        },
        "monitoring": {
            "enabled": True
        },
        "ssl": {
            "enabled": True
        },
        "scaling": {
            "horizontal": {
                "enabled": True,
                "min_replicas": 3,
                "max_replicas": 10
            }
        },
        "backup": {
            "enabled": True
        },
        "security": {
            "rate_limiting": {
                "enabled": True
            }
        }
    }
}

# Función para obtener configuración por entorno
def get_config(environment: str = "production") -> dict:
    """Obtener configuración por entorno"""
    base_config = deployment_config.copy()
    env_config = environments.get(environment, environments["production"])
    
    # Merge configurations
    def merge_dicts(base: dict, override: dict) -> dict:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge_dicts(base[key], value)
            else:
                base[key] = value
        return base
    
    return merge_dicts(base_config, env_config)

# Función para validar configuración
def validate_config(config: dict) -> bool:
    """Validar configuración"""
    required_fields = [
        "environment",
        "database.host",
        "database.port",
        "database.name",
        "redis.host",
        "redis.port"
    ]
    
    for field in required_fields:
        keys = field.split(".")
        current = config
        try:
            for key in keys:
                current = current[key]
            if current is None or current == "":
                print(f"Missing or empty field: {field}")
                return False
        except KeyError:
            print(f"Missing field: {field}")
            return False
    
    return True

# Función para generar archivo de configuración
def generate_config_file(environment: str = "production", output_file: str = "deployment_config.yaml"):
    """Generar archivo de configuración"""
    import yaml
    
    config = get_config(environment)
    
    if not validate_config(config):
        print("Configuration validation failed")
        return False
    
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration file generated: {output_file}")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate deployment configuration")
    parser.add_argument("--environment", choices=["development", "staging", "production"], 
                       default="production", help="Environment")
    parser.add_argument("--output", default="deployment_config.yaml", help="Output file")
    
    args = parser.parse_args()
    
    generate_config_file(args.environment, args.output)
