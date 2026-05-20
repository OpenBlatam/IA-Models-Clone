# CI/CD Integration Resources

# CodePipeline for automated deployments (optional)
resource "aws_codepipeline" "deploy" {
  count    = var.enable_codepipeline ? 1 : 0
  name     = "${var.project_name}-pipeline-${var.environment}"
  role_arn = aws_iam_role.codepipeline[0].arn

  artifact_store {
    location = aws_s3_bucket.codepipeline_artifacts[0].bucket
    type     = "S3"
  }

  stage {
    name = "Source"

    action {
      name             = "Source"
      category         = "Source"
      owner            = "AWS"
      provider         = "CodeStarSourceConnection"
      version          = "1"
      output_artifacts = ["source_output"]

      configuration = {
        ConnectionArn    = var.codestar_connection_arn
        FullRepositoryId = var.github_repository
        BranchName       = var.github_branch
      }
    }
  }

  stage {
    name = "Build"

    action {
      name             = "Build"
      category         = "Build"
      owner            = "AWS"
      provider         = "CodeBuild"
      input_artifacts  = ["source_output"]
      output_artifacts = ["build_output"]
      version         = "1"

      configuration = {
        ProjectName = aws_codebuild_project.build[0].name
      }
    }
  }

  stage {
    name = "Deploy"

    action {
      name            = "Deploy"
      category        = "Deploy"
      owner           = "AWS"
      provider        = "CodeDeployToECS"
      input_artifacts = ["build_output"]
      version         = "1"

      configuration = {
        ApplicationName     = aws_codedeploy_app.app[0].name
        DeploymentGroupName  = aws_codedeploy_deployment_group.app[0].deployment_group_name
      }
    }
  }
}

# CodeBuild project
resource "aws_codebuild_project" "build" {
  count        = var.enable_codepipeline ? 1 : 0
  name         = "${var.project_name}-build-${var.environment}"
  service_role = aws_iam_role.codebuild[0].arn

  artifacts {
    type = "CODEPIPELINE"
  }

  environment {
    compute_type                = "BUILD_GENERAL1_SMALL"
    image                       = "aws/codebuild/standard:5.0"
    type                        = "LINUX_CONTAINER"
    image_pull_credentials_type  = "CODEBUILD"
    privileged_mode             = true

    environment_variable {
      name  = "AWS_DEFAULT_REGION"
      value = var.aws_region
    }

    environment_variable {
      name  = "AWS_ACCOUNT_ID"
      value = data.aws_caller_identity.current.account_id
    }
  }

  source {
    type = "CODEPIPELINE"
    buildspec = "buildspec.yml"
  }
}

# S3 bucket for CodePipeline artifacts
resource "aws_s3_bucket" "codepipeline_artifacts" {
  count  = var.enable_codepipeline ? 1 : 0
  bucket = "${var.project_name}-codepipeline-artifacts-${var.environment}"

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-codepipeline-artifacts-${var.environment}"
    }
  )
}

# IAM roles for CI/CD
resource "aws_iam_role" "codepipeline" {
  count = var.enable_codepipeline ? 1 : 0
  name  = "${var.project_name}-codepipeline-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "codepipeline.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role" "codebuild" {
  count = var.enable_codepipeline ? 1 : 0
  name  = "${var.project_name}-codebuild-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "codebuild.amazonaws.com"
        }
      }
    ]
  })
}

# CodeDeploy application (if using)
resource "aws_codedeploy_app" "app" {
  count            = var.enable_codepipeline ? 1 : 0
  compute_platform = "ECS"
  name             = "${var.project_name}-${var.environment}"
}

resource "aws_codedeploy_deployment_group" "app" {
  count            = var.enable_codepipeline ? 1 : 0
  app_name         = aws_codedeploy_app.app[0].name
  deployment_group_name = "${var.project_name}-${var.environment}"

  service_role_arn = aws_iam_role.codedeploy[0].arn

  ecs_service {
    cluster_name = var.ecs_cluster_name != "" ? var.ecs_cluster_name : null
    service_name = var.ecs_service_name != "" ? var.ecs_service_name : null
  }

  auto_rollback_configuration {
    enabled = true
    events  = ["DEPLOYMENT_FAILURE"]
  }
}

resource "aws_iam_role" "codedeploy" {
  count = var.enable_codepipeline ? 1 : 0
  name  = "${var.project_name}-codedeploy-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "codedeploy.amazonaws.com"
        }
      }
    ]
  })
}

