import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export const cn = (...inputs: ClassValue[]): string => {
  return twMerge(clsx(inputs));
};

const DATE_FORMAT_OPTIONS: Intl.DateTimeFormatOptions = {
  year: 'numeric',
  month: 'short',
  day: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
};

const LOCALE = 'es-ES';

export const formatDate = (date: string | Date): string => {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  
  if (isNaN(dateObj.getTime())) {
    return 'Invalid date';
  }
  
  return new Intl.DateTimeFormat(LOCALE, DATE_FORMAT_OPTIONS).format(dateObj);
};

const MILLION = 1000000;
const THOUSAND = 1000;

export const formatNumber = (num: number): string => {
  if (num >= MILLION) {
    return `${(num / MILLION).toFixed(1)}M`;
  }
  
  if (num >= THOUSAND) {
    return `${(num / THOUSAND).toFixed(1)}K`;
  }
  
  return num.toString();
};

type PlatformName = 'tiktok' | 'instagram' | 'youtube';

const PLATFORM_COLORS: Record<PlatformName, string> = {
  tiktok: 'bg-black text-white',
  instagram: 'bg-gradient-to-r from-purple-500 to-pink-500 text-white',
  youtube: 'bg-red-600 text-white',
};

const DEFAULT_PLATFORM_COLOR = 'bg-gray-500 text-white';

export const getPlatformColor = (platform: string): string => {
  const normalizedPlatform = platform.toLowerCase() as PlatformName;
  return PLATFORM_COLORS[normalizedPlatform] || DEFAULT_PLATFORM_COLOR;
};

const PLATFORM_ICONS: Record<PlatformName, string> = {
  tiktok: '🎵',
  instagram: '📷',
  youtube: '▶️',
};

const DEFAULT_PLATFORM_ICON = '📱';

export const getPlatformIcon = (platform: string): string => {
  const normalizedPlatform = platform.toLowerCase() as PlatformName;
  return PLATFORM_ICONS[normalizedPlatform] || DEFAULT_PLATFORM_ICON;
};

export * from './utils/formatters';
export * from './utils/errorHandler';
export * from './utils/validators';
export * from './utils/helpers';
export * from './utils/formHelpers';
export * from './utils/performance';
export * from './utils/accessibility';
export * from './utils/arrayHelpers';
export * from './utils/objectHelpers';
export * from './utils/stringHelpers';
export * from './utils/numberHelpers';
export * from './utils/memoization';
export * from './utils/dateHelpers';
export * from './utils/colorHelpers';
export * from './utils/validationHelpers';
export * from './utils/storageHelpers';
export * from './utils/urlHelpers';
export * from './utils/cryptoHelpers';
export * from './utils/fileHelpers';
export * from './utils/domHelpers';
export * from './utils/eventHelpers';
export * from './utils/asyncHelpers';
export * from './utils/cacheHelpers';
export * from './utils/deviceHelpers';
export * from './utils/queryHelpers';
export * from './utils/analyticsHelpers';
export * from './utils/performanceHelpers';
export * from './utils/errorTracking';

