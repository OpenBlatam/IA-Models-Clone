// Input sanitization functions (pure functions)

export function sanitizeString(input: string): string {
  if (typeof input !== 'string') return '';
  
  return input
    .trim()
    .replace(/[<>]/g, '') // Remove potential HTML tags
    .replace(/javascript:/gi, '') // Remove javascript: protocol
    .replace(/on\w+=/gi, ''); // Remove event handlers
}

export function sanitizeEmail(email: string): string {
  if (typeof email !== 'string') return '';
  
  return sanitizeString(email).toLowerCase();
}

export function sanitizeNumber(input: string | number): number {
  if (typeof input === 'number') {
    return isNaN(input) ? 0 : input;
  }
  
  if (typeof input !== 'string') return 0;
  
  const cleaned = input.replace(/[^0-9.-]/g, '');
  const num = parseFloat(cleaned);
  
  return isNaN(num) ? 0 : num;
}

export function sanitizeUrl(url: string): string {
  if (typeof url !== 'string') return '';
  
  const sanitized = sanitizeString(url);
  
  // Basic URL validation
  try {
    const urlObj = new URL(sanitized);
    // Only allow http and https protocols
    if (urlObj.protocol === 'http:' || urlObj.protocol === 'https:') {
      return sanitized;
    }
  } catch {
    // If URL parsing fails, return empty string
    return '';
  }
  
  return '';
}

export function escapeHtml(text: string): string {
  if (typeof text !== 'string') return '';
  
  const map: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;',
  };
  
  return text.replace(/[&<>"']/g, (m) => map[m]);
}

