# PowerShell deployment script for Manuales Hogar AI on AWS

param(
    [string]$AwsRegion = "us-east-1",
    [string]$AwsAccountId = "",
    [string]$EcrRepository = "manuales-hogar-ai",
    [string]$ImageTag = "latest",
    [string]$Environment = "dev"
)

$ErrorActionPreference = "Stop"

# Configuration
$AppName = "manuales-hogar-ai"

# Colors for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

# Check required environment variables
function Check-Environment {
    Write-ColorOutput Yellow "Checking environment variables..."
    
    if ([string]::IsNullOrEmpty($AwsAccountId)) {
        Write-ColorOutput Red "Error: AWS_ACCOUNT_ID is not set"
        exit 1
    }
    
    if ([string]::IsNullOrEmpty($env:OPENROUTER_API_KEY)) {
        Write-ColorOutput Yellow "Warning: OPENROUTER_API_KEY is not set. It will need to be set in AWS Secrets Manager."
    }
    
    Write-ColorOutput Green "Environment check passed"
}

# Build Docker image
function Build-Image {
    Write-ColorOutput Yellow "Building Docker image..."
    $scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
    $projectRoot = Join-Path $scriptPath "..\..\"
    Set-Location $projectRoot
    
    docker build -t "${EcrRepository}:${ImageTag}" .
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "Docker build failed"
        exit 1
    }
    
    Write-ColorOutput Green "Docker image built successfully"
}

# Push to ECR
function Push-ToECR {
    Write-ColorOutput Yellow "Pushing image to ECR..."
    
    # Login to ECR
    $loginCommand = aws ecr get-login-password --region $AwsRegion | `
        docker login --username AWS --password-stdin "${AwsAccountId}.dkr.ecr.${AwsRegion}.amazonaws.com"
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "ECR login failed"
        exit 1
    }
    
    # Create ECR repository if it doesn't exist
    $repoExists = aws ecr describe-repositories --repository-names $EcrRepository --region $AwsRegion 2>$null
    if (-not $repoExists) {
        aws ecr create-repository --repository-name $EcrRepository --region $AwsRegion
    }
    
    # Tag and push
    $imageUri = "${AwsAccountId}.dkr.ecr.${AwsRegion}.amazonaws.com/${EcrRepository}:${ImageTag}"
    docker tag "${EcrRepository}:${ImageTag}" $imageUri
    docker push $imageUri
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "Docker push failed"
        exit 1
    }
    
    Write-ColorOutput Green "Image pushed to ECR successfully"
}

# Deploy CDK stack
function Deploy-CDK {
    Write-ColorOutput Yellow "Deploying CDK stack..."
    $scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
    $infraPath = Join-Path $scriptPath "..\infrastructure"
    Set-Location $infraPath
    
    # Install CDK dependencies
    if (-not (Test-Path ".venv")) {
        python -m venv .venv
    }
    & .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    
    # Bootstrap CDK (if needed)
    cdk bootstrap "aws://${AwsAccountId}/${AwsRegion}" 2>$null
    
    # Deploy
    $env:OPENROUTER_API_KEY = $env:OPENROUTER_API_KEY
    $env:DB_USERNAME = if ($env:DB_USERNAME) { $env:DB_USERNAME } else { "admin" }
    $env:DB_PASSWORD = $env:DB_PASSWORD
    $env:ENVIRONMENT = $Environment
    
    cdk deploy --require-approval never `
        --context "region=${AwsRegion}" `
        --context "account=${AwsAccountId}"
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "CDK deployment failed"
        exit 1
    }
    
    Write-ColorOutput Green "CDK stack deployed successfully"
}

# Update ECS service with new image
function Update-ECSService {
    Write-ColorOutput Yellow "Updating ECS service..."
    
    $clusterName = "${AppName}-cluster"
    $serviceName = "${AppName}-service"
    
    # Get task definition
    $taskDef = aws ecs describe-task-definition `
        --task-definition "${AppName}-task-def" `
        --region $AwsRegion `
        --query 'taskDefinition.taskDefinitionArn' `
        --output text
    
    # Update service
    aws ecs update-service `
        --cluster $clusterName `
        --service $serviceName `
        --task-definition $taskDef `
        --force-new-deployment `
        --region $AwsRegion | Out-Null
    
    Write-ColorOutput Green "ECS service update initiated"
    Write-ColorOutput Yellow "Waiting for service to stabilize..."
    
    aws ecs wait services-stable `
        --cluster $clusterName `
        --services $serviceName `
        --region $AwsRegion
    
    Write-ColorOutput Green "ECS service updated successfully"
}

# Main deployment flow
function Main {
    Write-ColorOutput Green "Starting deployment of ${AppName}..."
    
    Check-Environment
    Build-Image
    Push-ToECR
    Deploy-CDK
    Update-ECSService
    
    Write-ColorOutput Green "Deployment completed successfully!"
    Write-ColorOutput Yellow "Get the load balancer DNS from CDK outputs"
}

# Run main function
Main




