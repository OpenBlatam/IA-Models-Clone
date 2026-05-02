#!/bin/bash

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

sed "s/\${AWS_ACCOUNT_ID}/$AWS_ACCOUNT_ID/g" aws-task-definition.json > aws-task-definition-updated.json

echo "Task definition updated with AWS Account ID: $AWS_ACCOUNT_ID"
