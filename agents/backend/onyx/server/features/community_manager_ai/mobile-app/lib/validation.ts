import { z } from 'zod';

const VALIDATION_LIMITS = {
  POST_CONTENT_MIN: 1,
  POST_CONTENT_MAX: 5000,
  MEME_CAPTION_MAX: 500,
  TEMPLATE_NAME_MIN: 1,
  TEMPLATE_NAME_MAX: 100,
  TEMPLATE_CONTENT_MIN: 1,
  TEMPLATE_CONTENT_MAX: 10000,
  TAG_MAX_LENGTH: 50,
  MAX_TAGS: 10,
} as const;

const platformEnum = z.enum([
  'facebook',
  'instagram',
  'twitter',
  'linkedin',
  'tiktok',
  'youtube',
]);

const isoDateString = z.string().refine(
  (val) => {
    if (!val) return true;
    const date = new Date(val);
    return !isNaN(date.getTime());
  },
  { message: 'Invalid date' }
);

export const postSchema = z.object({
  content: z
    .string()
    .min(VALIDATION_LIMITS.POST_CONTENT_MIN, 'Content is required')
    .max(VALIDATION_LIMITS.POST_CONTENT_MAX, 'Content is too long'),
  platforms: z
    .array(platformEnum)
    .min(1, 'Select at least one platform')
    .max(6, 'Maximum 6 platforms'),
  scheduled_time: isoDateString.optional(),
  media_paths: z.array(z.string()).max(10, 'Maximum 10 media files').optional(),
  tags: z
    .array(z.string().max(VALIDATION_LIMITS.TAG_MAX_LENGTH, 'Tag is too long'))
    .max(VALIDATION_LIMITS.MAX_TAGS, `Maximum ${VALIDATION_LIMITS.MAX_TAGS} tags`)
    .optional(),
});

export const memeSchema = z.object({
  caption: z
    .string()
    .max(VALIDATION_LIMITS.MEME_CAPTION_MAX, 'Caption is too long')
    .optional(),
  tags: z
    .array(z.string().max(VALIDATION_LIMITS.TAG_MAX_LENGTH, 'Tag is too long'))
    .max(VALIDATION_LIMITS.MAX_TAGS, `Maximum ${VALIDATION_LIMITS.MAX_TAGS} tags`)
    .optional(),
  category: z.string().optional(),
});

export const templateSchema = z.object({
  name: z
    .string()
    .min(VALIDATION_LIMITS.TEMPLATE_NAME_MIN, 'Name is required')
    .max(VALIDATION_LIMITS.TEMPLATE_NAME_MAX, 'Name is too long')
    .trim(),
  content: z
    .string()
    .min(VALIDATION_LIMITS.TEMPLATE_CONTENT_MIN, 'Content is required')
    .max(VALIDATION_LIMITS.TEMPLATE_CONTENT_MAX, 'Content is too long'),
  platform: platformEnum.optional(),
  variables: z
    .array(z.string().max(VALIDATION_LIMITS.TAG_MAX_LENGTH, 'Variable name is too long'))
    .optional(),
  category: z.string().optional(),
});

export const platformConnectSchema = z.object({
  platform: platformEnum,
  credentials: z.record(z.string(), z.string()).refine(
    (credentials) => Object.keys(credentials).length > 0,
    { message: 'Credentials are required' }
  ),
});

export type PostFormData = z.infer<typeof postSchema>;
export type MemeFormData = z.infer<typeof memeSchema>;
export type TemplateFormData = z.infer<typeof templateSchema>;
export type PlatformConnectFormData = z.infer<typeof platformConnectSchema>;

