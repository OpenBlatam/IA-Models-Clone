import { Platform } from '@/types';
import { PLATFORM_OPTIONS } from '@/lib/constants';

export const getPlatformLabel = (platform: Platform): string => {
  const option = PLATFORM_OPTIONS.find((opt) => opt.value === platform);
  return option?.label || platform;
};

export const isValidPlatform = (value: string): value is Platform => {
  return PLATFORM_OPTIONS.some((opt) => opt.value === value);
};

export const getPlatformOptions = () => {
  return PLATFORM_OPTIONS.map((opt) => ({
    value: opt.value as Platform,
    label: opt.label,
  }));
};



