export const PLATFORMS = [
  { id: 'facebook', name: 'Facebook', icon: 'logo-facebook', color: '#1877f2' },
  { id: 'instagram', name: 'Instagram', icon: 'logo-instagram', color: '#e4405f' },
  { id: 'twitter', name: 'Twitter/X', icon: 'logo-twitter', color: '#1da1f2' },
  { id: 'linkedin', name: 'LinkedIn', icon: 'logo-linkedin', color: '#0077b5' },
  { id: 'tiktok', name: 'TikTok', icon: 'musical-notes', color: '#000000' },
  { id: 'youtube', name: 'YouTube', icon: 'logo-youtube', color: '#ff0000' },
] as const;

export const POST_STATUS = {
  SCHEDULED: 'scheduled',
  PUBLISHED: 'published',
  CANCELLED: 'cancelled',
} as const;

export const COLORS = {
  primary: '#0ea5e9',
  success: '#10b981',
  warning: '#f59e0b',
  error: '#ef4444',
  gray: {
    50: '#f9fafb',
    100: '#f3f4f6',
    200: '#e5e7eb',
    300: '#d1d5db',
    400: '#9ca3af',
    500: '#6b7280',
    600: '#4b5563',
    700: '#374151',
    800: '#1f2937',
    900: '#111827',
  },
} as const;


