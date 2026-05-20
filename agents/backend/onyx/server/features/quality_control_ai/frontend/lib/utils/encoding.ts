export const base64Encode = (str: string): string => {
  if (typeof window !== 'undefined' && window.btoa) {
    return window.btoa(str);
  }
  return Buffer.from(str).toString('base64');
};

export const base64Decode = (str: string): string => {
  if (typeof window !== 'undefined' && window.atob) {
    return window.atob(str);
  }
  return Buffer.from(str, 'base64').toString('utf-8');
};

export const urlEncode = (str: string): string => {
  return encodeURIComponent(str);
};

export const urlDecode = (str: string): string => {
  return decodeURIComponent(str);
};

export const htmlEncode = (str: string): string => {
  const map: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;',
  };
  return str.replace(/[&<>"']/g, (char) => map[char]);
};

export const htmlDecode = (str: string): string => {
  const map: Record<string, string> = {
    '&amp;': '&',
    '&lt;': '<',
    '&gt;': '>',
    '&quot;': '"',
    '&#039;': "'",
    '&#39;': "'",
  };
  return str.replace(/&amp;|&lt;|&gt;|&quot;|&#039;|&#39;/g, (entity) => map[entity]);
};

export const hexEncode = (str: string): string => {
  return str
    .split('')
    .map((char) => char.charCodeAt(0).toString(16).padStart(2, '0'))
    .join('');
};

export const hexDecode = (hex: string): string => {
  const bytes: number[] = [];
  for (let i = 0; i < hex.length; i += 2) {
    bytes.push(parseInt(hex.substr(i, 2), 16));
  }
  return String.fromCharCode(...bytes);
};

