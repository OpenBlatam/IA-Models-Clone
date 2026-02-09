export const ToastType = {
  SUCCESS: 'success',
  ERROR: 'error',
  INFO: 'info',
  WARNING: 'warning',
} as const;

export type ToastTypeValue = (typeof ToastType)[keyof typeof ToastType];

export const ButtonVariant = {
  PRIMARY: 'primary',
  SECONDARY: 'secondary',
  OUTLINE: 'outline',
  GHOST: 'ghost',
} as const;

export type ButtonVariantValue = (typeof ButtonVariant)[keyof typeof ButtonVariant];

export const ButtonSize = {
  SMALL: 'small',
  MEDIUM: 'medium',
  LARGE: 'large',
} as const;

export type ButtonSizeValue = (typeof ButtonSize)[keyof typeof ButtonSize];

export const LoadingSize = {
  SMALL: 'small',
  MEDIUM: 'medium',
  LARGE: 'large',
} as const;

export type LoadingSizeValue = (typeof LoadingSize)[keyof typeof LoadingSize];

