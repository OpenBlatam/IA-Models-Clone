#!/bin/bash

# AWS Configuration
AWS_REGION="us-east-1"
AWS_ACCESS_KEY_ID="AKIAZYJ6TBZ2WUTHGSTD"
AWS_SECRET_ACCESS_KEY="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"
BUCKET_NAME="blatamcursos"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Checking bucket contents...${NC}"

# List all objects in the bucket
aws s3 ls "s3://$BUCKET_NAME/" --region "$AWS_REGION" | while read -r line; do
    # Extract the file name from the line
    filename=$(echo "$line" | awk '{print $4}')
    
    # Check if it's a video file
    if [[ "$filename" == *.mp4 ]]; then
        echo -e "${YELLOW}Found video: $filename${NC}"
        
        # Get the URL for the file
        url="https://$BUCKET_NAME.s3.$AWS_REGION.amazonaws.com/$filename"
        echo -e "${GREEN}URL: $url${NC}"
        
        # Check if the file is publicly accessible
        if curl -s -I "$url" | grep -q "200 OK"; then
            echo -e "${GREEN}File is publicly accessible${NC}"
        else
            echo -e "${RED}File is not publicly accessible${NC}"
        fi
        echo "---"
    fi
done

echo -e "${GREEN}Check completed!${NC}"

# Show bucket policy
echo -e "\n${GREEN}Checking bucket policy...${NC}"
aws s3api get-bucket-policy --bucket "$BUCKET_NAME" --region "$AWS_REGION" || echo -e "${RED}No bucket policy found${NC}"

# Show bucket ACL
echo -e "\n${GREEN}Checking bucket ACL...${NC}"
aws s3api get-bucket-acl --bucket "$BUCKET_NAME" --region "$AWS_REGION" || echo -e "${RED}Failed to get bucket ACL${NC}" 