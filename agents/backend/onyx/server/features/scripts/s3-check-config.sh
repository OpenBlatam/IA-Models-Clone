#!/bin/bash

# Configuration
AWS_REGION=${AWS_REGION:-"us-east-1"}
VIDEO_BUCKET="blatamcursos"

echo "Checking S3 bucket configuration..."

# Check bucket policy
echo -e "\nBucket Policy:"
aws s3api get-bucket-policy --bucket "${VIDEO_BUCKET}" || echo "No bucket policy found"

# Check bucket ACL
echo -e "\nBucket ACL:"
aws s3api get-bucket-acl --bucket "${VIDEO_BUCKET}"

# Check public access block settings
echo -e "\nPublic Access Block Settings:"
aws s3api get-public-access-block --bucket "${VIDEO_BUCKET}"

# Check CORS configuration
echo -e "\nCORS Configuration:"
aws s3api get-bucket-cors --bucket "${VIDEO_BUCKET}" || echo "No CORS configuration found"

# List objects and their ACLs
echo -e "\nObject ACLs:"
aws s3api list-objects --bucket "${VIDEO_BUCKET}" --query 'Contents[].Key' --output text | while read -r key; do
    echo -e "\nChecking ACL for: $key"
    aws s3api get-object-acl --bucket "${VIDEO_BUCKET}" --key "$key"
done

echo -e "\nConfiguration check completed!" 