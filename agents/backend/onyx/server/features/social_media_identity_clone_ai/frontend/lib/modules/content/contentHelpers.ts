import { ContentType } from '@/types';
import { CONTENT_TYPE_OPTIONS } from '@/lib/constants';

export const getContentTypeLabel = (contentType: ContentType): string => {
  const option = CONTENT_TYPE_OPTIONS.find((opt) => opt.value === contentType);
  return option?.label || contentType;
};

export const isValidContentType = (value: string): value is ContentType => {
  return CONTENT_TYPE_OPTIONS.some((opt) => opt.value === value);
};

export const getContentTypeOptions = () => {
  return CONTENT_TYPE_OPTIONS.map((opt) => ({
    value: opt.value as ContentType,
    label: opt.label,
  }));
};



