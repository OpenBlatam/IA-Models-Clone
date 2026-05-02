#!/bin/bash

# Configuration
AWS_REGION=${AWS_REGION:-"us-east-1"}
VIDEO_BUCKET="blatamcursos"

# List all objects in the bucket
echo "Listing objects in bucket..."
objects=$(aws s3api list-objects --bucket "${VIDEO_BUCKET}" --query 'Contents[].Key' --output text)

# Set ACL for each object
for key in $objects; do
    echo "Setting ACL for: $key"
    aws s3api put-object-acl \
        --bucket "${VIDEO_BUCKET}" \
        --key "$key" \
        --acl public-read
done

echo "Video permissions updated successfully!" 