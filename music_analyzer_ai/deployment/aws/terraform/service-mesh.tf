# Service Mesh Configuration (App Mesh)
# Provides service discovery, load balancing, and traffic management

# App Mesh
resource "aws_appmesh_mesh" "main" {
  name = "music-analyzer-mesh-${var.environment}"

  spec {
    egress_filter {
      type = "ALLOW_ALL"
    }
  }

  tags = {
    Name = "music-analyzer-mesh-${var.environment}"
  }
}

# Virtual Node for Music Analyzer Service
resource "aws_appmesh_virtual_node" "music_analyzer" {
  name      = "music-analyzer-node-${var.environment}"
  mesh_name = aws_appmesh_mesh.main.name

  spec {
    listener {
      port_mapping {
        port     = 8010
        protocol = "http"
      }

      health_check {
        protocol            = "http"
        path                = "/health"
        healthy_threshold   = 2
        unhealthy_threshold = 3
        timeout_millis      = 2000
        interval_millis     = 5000
      }
    }

    service_discovery {
      aws_cloud_map {
        namespace_name = aws_service_discovery_private_dns_namespace.main.name
        service_name   = aws_appmesh_virtual_service.music_analyzer.name
      }
    }

    backend {
      virtual_service {
        virtual_service_name = aws_appmesh_virtual_service.redis.name
      }
    }
  }
}

# Virtual Service
resource "aws_appmesh_virtual_service" "music_analyzer" {
  name      = "music-analyzer.${aws_service_discovery_private_dns_namespace.main.name}"
  mesh_name = aws_appmesh_mesh.main.name

  spec {
    provider {
      virtual_node {
        virtual_node_name = aws_appmesh_virtual_node.music_analyzer.name
      }
    }
  }
}

# Virtual Node for Redis
resource "aws_appmesh_virtual_node" "redis" {
  name      = "redis-node-${var.environment}"
  mesh_name = aws_appmesh_mesh.main.name

  spec {
    listener {
      port_mapping {
        port     = 6379
        protocol = "tcp"
      }
    }

    service_discovery {
      dns {
        hostname = aws_elasticache_replication_group.redis.configuration_endpoint_address
      }
    }
  }
}

# Virtual Service for Redis
resource "aws_appmesh_virtual_service" "redis" {
  name      = "redis.${aws_service_discovery_private_dns_namespace.main.name}"
  mesh_name = aws_appmesh_mesh.main.name

  spec {
    provider {
      virtual_node {
        virtual_node_name = aws_appmesh_virtual_node.redis.name
      }
    }
  }
}

# Service Discovery
resource "aws_service_discovery_private_dns_namespace" "main" {
  name        = "music-analyzer.local"
  description = "Service discovery namespace for Music Analyzer AI"
  vpc         = aws_vpc.main.id
}

resource "aws_service_discovery_service" "music_analyzer" {
  name = "music-analyzer"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id

    dns_records {
      ttl  = 10
      type = "A"
    }

    routing_policy = "MULTIVALUE"
  }

  health_check_grace_period_seconds = 30
}

# Virtual Router
resource "aws_appmesh_virtual_router" "main" {
  name      = "music-analyzer-router-${var.environment}"
  mesh_name = aws_appmesh_mesh.main.name

  spec {
    listener {
      port_mapping {
        port     = 8010
        protocol = "http"
      }
    }
  }
}

# Route Configuration
resource "aws_appmesh_route" "main" {
  name                = "music-analyzer-route-${var.environment}"
  mesh_name           = aws_appmesh_mesh.main.name
  virtual_router_name = aws_appmesh_virtual_router.main.name

  spec {
    http_route {
      match {
        prefix = "/"
      }

      action {
        weighted_target {
          virtual_node = aws_appmesh_virtual_node.music_analyzer.name
          weight       = 100
        }
      }

      retry_policy {
        http_retry_events = ["server-error", "client-error", "gateway-error"]
        max_retries       = 3
        per_retry_timeout {
          unit  = "ms"
          value = 1000
        }
      }

      timeout {
        idle {
          unit  = "ms"
          value = 5000
        }
        per_request {
          unit  = "ms"
          value = 30000
        }
      }
    }
  }
}

# IAM Role for App Mesh
resource "aws_iam_role" "appmesh" {
  name = "music-analyzer-appmesh-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "appmesh.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "appmesh" {
  name = "music-analyzer-appmesh-policy-${var.environment}"
  role = aws_iam_role.appmesh.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "appmesh:*",
          "servicediscovery:*"
        ]
        Resource = "*"
      }
    ]
  })
}




