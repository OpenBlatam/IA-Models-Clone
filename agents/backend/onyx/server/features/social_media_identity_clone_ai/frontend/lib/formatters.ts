export const formatKeyToLabel = (key: string): string => {
  return key
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

export const formatPlatformName = (platform: string): string => {
  return platform.charAt(0).toUpperCase() + platform.slice(1).toLowerCase();
};

export const formatContentType = (contentType: string): string => {
  return contentType.charAt(0).toUpperCase() + contentType.slice(1).toLowerCase();
};



