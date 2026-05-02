#!/bin/bash

# AWS Configuration
AWS_REGION="us-east-1"
AWS_ACCESS_KEY_ID="AKIAZYJ6TBZ2WUTHGSTD"
AWS_SECRET_ACCESS_KEY="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"
BUCKET_NAME="blatamcursos"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to upload a video
upload_video() {
    local video_path="$1"
    local video_name=$(basename "$video_path")
    
    echo -e "${GREEN}Uploading $video_name...${NC}"
    
    # Upload to S3 with public read access
    aws s3 cp "$video_path" "s3://$BUCKET_NAME/$video_name" \
        --region "$AWS_REGION" \
        --acl public-read \
        --content-type "video/mp4"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully uploaded $video_name${NC}"
    else
        echo -e "${RED}Failed to upload $video_name${NC}"
    fi
}

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if videos directory exists
VIDEOS_DIR="videos"
if [ ! -d "$VIDEOS_DIR" ]; then
    echo -e "${RED}Videos directory not found. Creating it...${NC}"
    mkdir -p "$VIDEOS_DIR"
fi

# Upload all videos in the directory
echo -e "${GREEN}Starting video upload process...${NC}"

# Upload videos 1-12 (IA Generativa Básico)
for i in {1..12}; do
    video_path="$VIDEOS_DIR/$i. IA Generativa.mp4"
    if [ -f "$video_path" ]; then
        upload_video "$video_path"
    else
        echo -e "${RED}Video not found: $video_path${NC}"
    fi
done

# Upload videos 13-24 (IA Generativa Avanzado)
for i in {13..24}; do
    video_path="$VIDEOS_DIR/$i. IA Generativa.mp4"
    if [ -f "$video_path" ]; then
        upload_video "$video_path"
    else
        echo -e "${RED}Video not found: $video_path${NC}"
    fi
done

echo -e "${GREEN}Upload process completed!${NC}"

# List all videos in the bucket
echo -e "${GREEN}Listing all videos in the bucket:${NC}"
aws s3 ls "s3://$BUCKET_NAME/" --region "$AWS_REGION" 