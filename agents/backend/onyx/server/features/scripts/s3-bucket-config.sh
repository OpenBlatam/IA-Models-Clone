#!/bin/bash

# Configuration
AWS_REGION=${AWS_REGION:-"us-east-1"}
VIDEO_BUCKET="blatamcursos"

echo "Configuring S3 bucket..."

# Disable all public access blocks
echo "Disabling public access blocks..."
aws s3api put-public-access-block \
    --bucket "${VIDEO_BUCKET}" \
    --public-access-block-configuration "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"

# Create bucket policy for public read access
echo "Creating bucket policy..."
aws s3api put-bucket-policy \
    --bucket "${VIDEO_BUCKET}" \
    --policy '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "PublicReadGetObject",
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:GetObject",
                "Resource": [
                    "arn:aws:s3:::'${VIDEO_BUCKET}'/*",
                    "arn:aws:s3:::'${VIDEO_BUCKET}'/Cursos/*"
                ]
            }
        ]
    }'

# Set CORS configuration
echo "Setting CORS configuration..."
aws s3api put-bucket-cors \
    --bucket "${VIDEO_BUCKET}" \
    --cors-configuration '{
        "CORSRules": [
            {
                "AllowedHeaders": ["*"],
                "AllowedMethods": ["GET", "HEAD"],
                "AllowedOrigins": ["*"],
                "ExposeHeaders": []
            }
        ]
    }'

# Set bucket ACL to public-read
echo "Setting bucket ACL..."
aws s3api put-bucket-acl \
    --bucket "${VIDEO_BUCKET}" \
    --acl public-read

echo "S3 bucket configuration completed!" 