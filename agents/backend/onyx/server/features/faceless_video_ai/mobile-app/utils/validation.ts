import { z } from 'zod';

// Auth Schemas
export const loginSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(6, 'Password must be at least 6 characters'),
});

export const registerSchema = z
  .object({
    email: z.string().email('Invalid email address'),
    password: z.string().min(6, 'Password must be at least 6 characters'),
    confirmPassword: z.string(),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "Passwords don't match",
    path: ['confirmPassword'],
  });

// Video Generation Schema
export const videoGenerationSchema = z.object({
  script: z.object({
    text: z.string().min(10, 'Script must be at least 10 characters'),
    language: z.string().length(2).optional(),
  }),
  video_config: z
    .object({
      resolution: z.string().regex(/^\d+x\d+$/).optional(),
      fps: z.number().min(24).max(60).optional(),
      style: z.enum(['realistic', 'animated', 'abstract', 'minimalist', 'dynamic']),
    })
    .optional(),
  audio_config: z
    .object({
      voice: z.enum(['male_1', 'male_2', 'female_1', 'female_2', 'neutral']),
      speed: z.number().min(0.5).max(2.0).optional(),
    })
    .optional(),
  subtitle_config: z
    .object({
      enabled: z.boolean(),
      style: z.enum(['simple', 'modern', 'bold', 'elegant', 'minimal']).optional(),
    })
    .optional(),
});

// Helper function to validate and extract errors
export function validateForm<T>(
  schema: z.ZodSchema<T>,
  data: unknown
): { success: true; data: T } | { success: false; errors: Record<string, string> } {
  try {
    const validated = schema.parse(data);
    return { success: true, data: validated };
  } catch (error) {
    if (error instanceof z.ZodError) {
      const errors: Record<string, string> = {};
      error.errors.forEach((err) => {
        if (err.path[0]) {
          errors[err.path[0] as string] = err.message;
        }
      });
      return { success: false, errors };
    }
    return { success: false, errors: { _: 'Validation failed' } };
  }
}


