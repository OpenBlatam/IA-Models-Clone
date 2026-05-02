#!/bin/bash

# Configuration
AWS_REGION=${AWS_REGION:-"us-east-1"}
GENERAL_BUCKET=${AWS_S3_BUCKET_NAME}
VIDEO_BUCKET=${AWS_S3_VIDEO_BUCKET_NAME}
CLIENT_BUCKET=${AWS_S3_CLIENT_BUCKET_NAME}
AI_BUCKET=${AWS_S3_AI_BUCKET_NAME}

# Function to create lifecycle configuration
create_lifecycle_config() {
  local bucket=$1
  local config_file=$2
  
  echo "Configuring lifecycle rules for bucket: $bucket"
  aws s3api put-bucket-lifecycle-configuration \
    --bucket "$bucket" \
    --lifecycle-configuration "file://$config_file"
}

# General bucket lifecycle configuration (for images, documents, etc.)
cat > general-bucket-lifecycle.json << EOF
{
  "Rules": [
    {
      "ID": "blatam-images",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "images/"
      },
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "NoncurrentVersionTransitions": [
        {
          "NoncurrentDays": 30,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 90
      },
      "AbortIncompleteMultipartUpload": {
        "DaysAfterInitiation": 7
      }
    },
    {
      "ID": "blatam-documents",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "documents/"
      },
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "NoncurrentVersionTransitions": [
        {
          "NoncurrentDays": 30,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 90
      },
      "AbortIncompleteMultipartUpload": {
        "DaysAfterInitiation": 7
      }
    }
  ]
}
EOF

# Video bucket lifecycle configuration
cat > video-bucket-lifecycle.json << EOF
{
  "Rules": [
    {
      "ID": "blatam-course-videos",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "courses/"
      },
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "NoncurrentVersionTransitions": [
        {
          "NoncurrentDays": 30,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 90
      },
      "AbortIncompleteMultipartUpload": {
        "DaysAfterInitiation": 7
      }
    },
    {
      "ID": "blatam-thumbnails",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "thumbnails/"
      },
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "NoncurrentVersionTransitions": [
        {
          "NoncurrentDays": 30,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 90
      },
      "AbortIncompleteMultipartUpload": {
        "DaysAfterInitiation": 7
      }
    }
  ]
}
EOF

# Client bucket lifecycle configuration
cat > client-bucket-lifecycle.json << EOF
{
  "Rules": [
    {
      "ID": "blatam-client-data",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "client-data/"
      },
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "NoncurrentVersionTransitions": [
        {
          "NoncurrentDays": 30,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 90
      },
      "AbortIncompleteMultipartUpload": {
        "DaysAfterInitiation": 7
      }
    },
    {
      "ID": "blatam-user-uploads",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "user-uploads/"
      },
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "NoncurrentVersionTransitions": [
        {
          "NoncurrentDays": 30,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 90
      },
      "AbortIncompleteMultipartUpload": {
        "DaysAfterInitiation": 7
      }
    }
  ]
}
EOF

# AI bucket lifecycle configuration
cat > ai-bucket-lifecycle.json << EOF
{
  "Rules": [
    {
      "ID": "blatam-ai-models",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "models/"
      },
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "NoncurrentVersionTransitions": [
        {
          "NoncurrentDays": 30,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 90
      },
      "AbortIncompleteMultipartUpload": {
        "DaysAfterInitiation": 7
      }
    },
    {
      "ID": "blatam-ai-data",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "data/"
      },
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "NoncurrentVersionTransitions": [
        {
          "NoncurrentDays": 30,
          "StorageClass": "STANDARD_IA"
        }
      ],
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 90
      },
      "AbortIncompleteMultipartUpload": {
        "DaysAfterInitiation": 7
      }
    }
  ]
}
EOF

# Apply lifecycle configurations
if [ ! -z "$GENERAL_BUCKET" ]; then
  create_lifecycle_config "$GENERAL_BUCKET" "general-bucket-lifecycle.json"
fi

if [ ! -z "$VIDEO_BUCKET" ]; then
  create_lifecycle_config "$VIDEO_BUCKET" "video-bucket-lifecycle.json"
fi

if [ ! -z "$CLIENT_BUCKET" ]; then
  create_lifecycle_config "$CLIENT_BUCKET" "client-bucket-lifecycle.json"
fi

if [ ! -z "$AI_BUCKET" ]; then
  create_lifecycle_config "$AI_BUCKET" "ai-bucket-lifecycle.json"
fi

# Cleanup temporary files
rm -f *-bucket-lifecycle.json

echo "S3 lifecycle configurations completed successfully!" 