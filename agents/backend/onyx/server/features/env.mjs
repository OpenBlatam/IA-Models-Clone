import { createEnv } from "@t3-oss/env-nextjs";
import { z } from "zod";

const isDevelopment = process.env.NODE_ENV === "development";

export const env = createEnv({
  server: {
    AUTH_SECRET: isDevelopment 
      ? z.string().default("development-secret")
      : z.string().min(1, "AUTH_SECRET is required"),
    GOOGLE_CLIENT_ID: isDevelopment
      ? z.string().default("development-client-id")
      : z.string().min(1, "GOOGLE_CLIENT_ID is required"),
    GOOGLE_CLIENT_SECRET: isDevelopment
      ? z.string().default("development-client-secret")
      : z.string().min(1, "GOOGLE_CLIENT_SECRET is required"),
    DATABASE_URL: isDevelopment
      ? z.string().default("postgresql://postgres:postgres@localhost:5432/postgres")
      : z.string().min(1, "DATABASE_URL is required"),
    // AWS Configuration
    AWS_REGION: z.string().default("us-east-1"),
    AWS_ACCESS_KEY_ID: z.string().min(1, "AWS_ACCESS_KEY_ID is required"),
    AWS_SECRET_ACCESS_KEY: z.string().min(1, "AWS_SECRET_ACCESS_KEY is required"),
    AWS_S3_VIDEO_BUCKET_NAME: z.string().default("blatamcursos"),
  },
  client: {
    NEXT_PUBLIC_APP_URL: z.string().url().default("http://localhost:3000"),
    NEXT_PUBLIC_AWS_REGION: z.string().default("us-east-1"),
    NEXT_PUBLIC_AWS_S3_VIDEO_BUCKET_NAME: z.string().default("blatamcursos"),
    NEXT_PUBLIC_API_URL: z.string().url().default("http://localhost:8001"),
  },
  runtimeEnv: {
    AUTH_SECRET: process.env.AUTH_SECRET,
    GOOGLE_CLIENT_ID: process.env.GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET: process.env.GOOGLE_CLIENT_SECRET,
    DATABASE_URL: process.env.DATABASE_URL,
    NEXT_PUBLIC_APP_URL: process.env.NEXT_PUBLIC_APP_URL,
    // AWS Configuration
    AWS_REGION: process.env.AWS_REGION,
    AWS_ACCESS_KEY_ID: process.env.AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY: process.env.AWS_SECRET_ACCESS_KEY,
    AWS_S3_VIDEO_BUCKET_NAME: process.env.AWS_S3_VIDEO_BUCKET_NAME,
    NEXT_PUBLIC_AWS_REGION: process.env.AWS_REGION,
    NEXT_PUBLIC_AWS_S3_VIDEO_BUCKET_NAME: process.env.AWS_S3_VIDEO_BUCKET_NAME,
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
  },
  skipValidation: !!process.env.SKIP_ENV_VALIDATION,
});
