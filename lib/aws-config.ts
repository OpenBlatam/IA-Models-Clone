import { S3Client } from '@aws-sdk/client-s3';

// Server-side S3 client
export const s3Client = new S3Client({
  region: process.env.AWS_REGION || 'us-east-1',
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  },
});

// Bucket names
export const S3_BUCKET_NAME = process.env.AWS_S3_BUCKET_NAME!;
export const S3_CLIENT_BUCKET_NAME = process.env.AWS_S3_CLIENT_BUCKET_NAME!;

// S3 URLs
export const S3_BUCKET_URL = `https://${S3_BUCKET_NAME}.s3.amazonaws.com`;
export const S3_CLIENT_BUCKET_URL = `https://${S3_CLIENT_BUCKET_NAME}.s3.amazonaws.com`; 