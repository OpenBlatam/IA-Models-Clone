export const sanitizeHtml = (html: string): string => {
  const div = document.createElement('div');
  div.textContent = html;
  return div.innerHTML;
};

export const sanitizeUrl = (url: string): string => {
  try {
    const urlObj = new URL(url);
    // Only allow http and https protocols
    if (urlObj.protocol === 'http:' || urlObj.protocol === 'https:') {
      return urlObj.toString();
    }
    return '';
  } catch {
    return '';
  }
};

export const sanitizeFilename = (filename: string): string => {
  return filename
    .replace(/[<>:"/\\|?*]/g, '')
    .replace(/\s+/g, '_')
    .replace(/^\.+|\.+$/g, '')
    .substring(0, 255);
};

export const sanitizeInput = (input: string): string => {
  return input
    .trim()
    .replace(/[<>]/g, '')
    .replace(/\s+/g, ' ');
};

export const escapeRegex = (str: string): string => {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
};

export const unescapeRegex = (str: string): string => {
  return str.replace(/\\(.)/g, '$1');
};

