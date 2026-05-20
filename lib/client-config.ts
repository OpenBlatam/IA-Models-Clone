import { S3Client } from "@aws-sdk/client-s3";

// Client-side configuration
export const S3_VIDEO_BUCKET_NAME = process.env.NEXT_PUBLIC_AWS_S3_VIDEO_BUCKET_NAME || 'blatamcursos';
export const AWS_REGION = process.env.NEXT_PUBLIC_AWS_REGION || 'us-east-1';
export const S3_VIDEO_BUCKET_URL = `https://${S3_VIDEO_BUCKET_NAME}.s3.${AWS_REGION}.amazonaws.com`;

// AWS Client Configuration
export const AWS_ACCESS_KEY_ID = process.env.NEXT_PUBLIC_AWS_ACCESS_KEY_ID;
export const AWS_SECRET_ACCESS_KEY = process.env.NEXT_PUBLIC_AWS_SECRET_ACCESS_KEY;

// Create S3 client
export const s3Client = new S3Client({
  region: AWS_REGION,
  credentials: {
    accessKeyId: AWS_ACCESS_KEY_ID!,
    secretAccessKey: AWS_SECRET_ACCESS_KEY!,
  },
});

// Helper function to create safe video URLs
export const createVideoUrl = (filename: string): string => {
  try {
    // Clean and format the filename to match bucket format
    const cleanFilename = filename
      .trim()
      .replace(/[^a-zA-Z0-9\s.-]/g, "") // Remove special characters
      .replace(/\s+/g, " ") // Replace multiple spaces with single space
      .trim();

    // Construct the video URL
    const videoUrl = `https://${S3_VIDEO_BUCKET_NAME}.s3.${AWS_REGION}.amazonaws.com/${cleanFilename}.mp4`;
    
    return videoUrl;
  } catch (error) {
    console.error("Error creating video URL:", error);
    return "";
  }
}; 