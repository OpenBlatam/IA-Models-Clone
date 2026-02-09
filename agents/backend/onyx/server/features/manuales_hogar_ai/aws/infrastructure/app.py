"""
AWS CDK Infrastructure for Manuales Hogar AI
============================================

This CDK app creates the complete AWS infrastructure for the Manuales Hogar AI service:
- VPC with public and private subnets
- RDS PostgreSQL database
- ECS Fargate cluster
- Application Load Balancer
- CloudWatch logging and monitoring
- Secrets Manager for API keys
- Auto-scaling configuration
"""

import os
from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput,
    Tags,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_rds as rds,
    aws_secretsmanager as secretsmanager,
    aws_logs as logs,
    aws_iam as iam,
    aws_elasticloadbalancingv2 as elbv2,
    aws_applicationautoscaling as autoscaling,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    aws_sns as sns,
)
from constructs import Construct


class ManualesHogarAIStack(Stack):
    """Main stack for Manuales Hogar AI infrastructure."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Configuration
        app_name = "manuales-hogar-ai"
        vpc_cidr = "10.0.0.0/16"
        
        # Environment variables
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        db_username = os.getenv("DB_USERNAME", "admin")
        db_password = os.getenv("DB_PASSWORD", "")

        # Create VPC
        vpc = ec2.Vpc(
            self,
            f"{app_name}-vpc",
            max_azs=2,
            cidr=vpc_cidr,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    subnet_type=ec2.SubnetType.PUBLIC,
                    name="Public",
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    name="Private",
                    cidr_mask=24,
                ),
            ],
            nat_gateways=1,
        )

        # Create Security Groups
        alb_sg = ec2.SecurityGroup(
            self,
            f"{app_name}-alb-sg",
            vpc=vpc,
            description="Security group for Application Load Balancer",
            allow_all_outbound=True,
        )
        alb_sg.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(80),
            "Allow HTTP from anywhere",
        )
        alb_sg.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(443),
            "Allow HTTPS from anywhere",
        )

        ecs_sg = ec2.SecurityGroup(
            self,
            f"{app_name}-ecs-sg",
            vpc=vpc,
            description="Security group for ECS tasks",
            allow_all_outbound=True,
        )
        ecs_sg.add_ingress_rule(
            alb_sg,
            ec2.Port.tcp(8000),
            "Allow traffic from ALB",
        )

        db_sg = ec2.SecurityGroup(
            self,
            f"{app_name}-db-sg",
            vpc=vpc,
            description="Security group for RDS database",
            allow_all_outbound=False,
        )
        db_sg.add_ingress_rule(
            ecs_sg,
            ec2.Port.tcp(5432),
            "Allow PostgreSQL from ECS tasks",
        )

        # Create Secrets Manager secret for OpenRouter API key
        openrouter_secret = secretsmanager.Secret(
            self,
            f"{app_name}-openrouter-secret",
            secret_name=f"{app_name}/openrouter-api-key",
            description="OpenRouter API Key for Manuales Hogar AI",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template='{"api_key": ""}',
                generate_string_key="api_key",
                exclude_characters='"@/\\',
            ) if not openrouter_api_key else None,
            secret_string_beta1=secretsmanager.SecretValueBeta1.from_unsafe_plaintext(
                f'{{"api_key": "{openrouter_api_key}"}}'
            ) if openrouter_api_key else None,
        )

        # Create RDS PostgreSQL database
        db_subnet_group = rds.SubnetGroup(
            self,
            f"{app_name}-db-subnet-group",
            vpc=vpc,
            description="Subnet group for RDS database",
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
        )

        # Create database with auto-generated credentials
        database = rds.DatabaseInstance(
            self,
            f"{app_name}-database",
            engine=rds.DatabaseInstanceEngine.postgres(
                version=rds.PostgresEngineVersion.VER_15_4
            ),
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.T3, ec2.InstanceSize.MICRO
            ),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            security_groups=[db_sg],
            subnet_group=db_subnet_group,
            database_name="manuales_hogar",
            credentials=rds.Credentials.from_generated_secret(
                db_username,
                secret_name=f"{app_name}/db-credentials",
            ),
            allocated_storage=20,
            max_allocated_storage=100,
            storage_encrypted=True,
            backup_retention=Duration.days(7),
            deletion_protection=False,
            removal_policy=RemovalPolicy.DESTROY,  # Change to RETAIN for production
            multi_az=False,  # Set to True for production
        )

        # Create CloudWatch Log Group
        log_group = logs.LogGroup(
            self,
            f"{app_name}-logs",
            log_group_name=f"/ecs/{app_name}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Create ECS Cluster
        cluster = ecs.Cluster(
            self,
            f"{app_name}-cluster",
            vpc=vpc,
            cluster_name=f"{app_name}-cluster",
            container_insights=True,
        )

        # Create ECS Task Definition
        task_definition = ecs.FargateTaskDefinition(
            self,
            f"{app_name}-task-def",
            memory_limit_mib=2048,
            cpu=1024,
        )

        # Get ECR image URI from environment or use placeholder
        ecr_repo = os.getenv("ECR_REPOSITORY", app_name)
        image_tag = os.getenv("IMAGE_TAG", "latest")
        aws_account_id = os.getenv("AWS_ACCOUNT_ID", self.account)
        aws_region = os.getenv("AWS_REGION", self.region)
        image_uri = f"{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{ecr_repo}:{image_tag}"
        
        # Add container to task definition
        container = task_definition.add_container(
            f"{app_name}-container",
            image=ecs.ContainerImage.from_registry(image_uri),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix=f"{app_name}",
                log_group=log_group,
            ),
            environment={
                "PORT": "8000",
                "DB_HOST": database.instance_endpoint.hostname,
                "DB_PORT": "5432",
                "DB_NAME": "manuales_hogar",
            },
            secrets={
                "OPENROUTER_API_KEY": ecs.Secret.from_secrets_manager(
                    openrouter_secret, "api_key"
                ),
                "DB_USER": ecs.Secret.from_secrets_manager(
                    database.secret, "username"
                ),
                "DB_PASSWORD": ecs.Secret.from_secrets_manager(
                    database.secret, "password"
                ),
            },
        )

        container.add_port_mappings(
            ecs.PortMapping(
                container_port=8000,
                protocol=ecs.Protocol.TCP,
            )
        )

        # Create ECS Service with Application Load Balancer
        service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            f"{app_name}-service",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=2,
            public_load_balancer=True,
            listener_port=80,
            service_name=f"{app_name}-service",
            health_check_grace_period=Duration.seconds(60),
        )

        # Configure health check
        service.target_group.configure_health_check(
            path="/api/v1/health",
            interval=Duration.seconds(30),
            timeout=Duration.seconds(5),
            healthy_threshold_count=2,
            unhealthy_threshold_count=3,
        )

        # Auto-scaling configuration
        scalable_target = service.service.auto_scale_task_count(
            min_capacity=2,
            max_capacity=10,
        )

        # Scale based on CPU utilization
        scalable_target.scale_on_cpu_utilization(
            "CpuScaling",
            target_utilization_percent=70,
            scale_in_cooldown=Duration.seconds(60),
            scale_out_cooldown=Duration.seconds(60),
        )

        # Scale based on memory utilization
        scalable_target.scale_on_memory_utilization(
            "MemoryScaling",
            target_utilization_percent=80,
            scale_in_cooldown=Duration.seconds(60),
            scale_out_cooldown=Duration.seconds(60),
        )

        # Create SNS topic for alerts
        alert_topic = sns.Topic(
            self,
            f"{app_name}-alerts",
            topic_name=f"{app_name}-alerts",
        )

        # CloudWatch Alarms
        # High CPU alarm
        cpu_alarm = cloudwatch.Alarm(
            self,
            f"{app_name}-high-cpu",
            metric=service.service.metric_cpu_utilization(),
            threshold=80,
            evaluation_periods=2,
            alarm_description="Alert when CPU utilization exceeds 80%",
        )
        cpu_alarm.add_alarm_action(cw_actions.SnsAction(alert_topic))

        # High memory alarm
        memory_alarm = cloudwatch.Alarm(
            self,
            f"{app_name}-high-memory",
            metric=service.service.metric_memory_utilization(),
            threshold=85,
            evaluation_periods=2,
            alarm_description="Alert when memory utilization exceeds 85%",
        )
        memory_alarm.add_alarm_action(cw_actions.SnsAction(alert_topic))

        # Database connection alarm
        db_connections_alarm = cloudwatch.Alarm(
            self,
            f"{app_name}-db-connections",
            metric=database.metric_database_connections(),
            threshold=80,
            evaluation_periods=2,
            alarm_description="Alert when database connections exceed 80%",
        )
        db_connections_alarm.add_alarm_action(cw_actions.SnsAction(alert_topic))

        # Outputs
        CfnOutput(
            self,
            "LoadBalancerDNS",
            value=service.load_balancer.load_balancer_dns_name,
            description="DNS name of the load balancer",
        )

        CfnOutput(
            self,
            "DatabaseEndpoint",
            value=database.instance_endpoint.hostname,
            description="RDS PostgreSQL endpoint",
        )

        CfnOutput(
            self,
            "DatabasePort",
            value=str(database.instance_endpoint.port),
            description="RDS PostgreSQL port",
        )

        CfnOutput(
            self,
            "OpenRouterSecretArn",
            value=openrouter_secret.secret_arn,
            description="ARN of OpenRouter API key secret",
        )

        CfnOutput(
            self,
            "ClusterName",
            value=cluster.cluster_name,
            description="ECS Cluster name",
        )

        CfnOutput(
            self,
            "ServiceName",
            value=service.service.service_name,
            description="ECS Service name",
        )

        # Add tags
        Tags.of(self).add("Application", app_name)
        Tags.of(self).add("Environment", os.getenv("ENVIRONMENT", "dev"))
        Tags.of(self).add("ManagedBy", "CDK")

