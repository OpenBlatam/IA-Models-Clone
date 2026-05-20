/**
 * Device detection utilities
 */

// Check if mobile
export function isMobile(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }

  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent
  );
}

// Check if tablet
export function isTablet(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }

  return /iPad|Android/i.test(navigator.userAgent) && window.innerWidth >= 768;
}

// Check if desktop
export function isDesktop(): boolean {
  return !isMobile() && !isTablet();
}

// Check if iOS
export function isIOS(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }

  return /iPad|iPhone|iPod/.test(navigator.userAgent);
}

// Check if Android
export function isAndroid(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }

  return /Android/i.test(navigator.userAgent);
}

// Check if touch device
export function isTouchDevice(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }

  return (
    'ontouchstart' in window ||
    navigator.maxTouchPoints > 0 ||
    (navigator as any).msMaxTouchPoints > 0
  );
}

// Get device type
export function getDeviceType(): 'mobile' | 'tablet' | 'desktop' {
  if (isMobile()) return 'mobile';
  if (isTablet()) return 'tablet';
  return 'desktop';
}

// Get browser info
export function getBrowserInfo(): {
  name: string;
  version: string;
  os: string;
} {
  if (typeof window === 'undefined') {
    return { name: 'unknown', version: 'unknown', os: 'unknown' };
  }

  const ua = navigator.userAgent;
  let browserName = 'unknown';
  let browserVersion = 'unknown';
  let os = 'unknown';

  // Browser detection
  if (ua.indexOf('Chrome') > -1 && ua.indexOf('Edg') === -1) {
    browserName = 'Chrome';
    const match = ua.match(/Chrome\/(\d+)/);
    browserVersion = match ? match[1] : 'unknown';
  } else if (ua.indexOf('Firefox') > -1) {
    browserName = 'Firefox';
    const match = ua.match(/Firefox\/(\d+)/);
    browserVersion = match ? match[1] : 'unknown';
  } else if (ua.indexOf('Safari') > -1 && ua.indexOf('Chrome') === -1) {
    browserName = 'Safari';
    const match = ua.match(/Version\/(\d+)/);
    browserVersion = match ? match[1] : 'unknown';
  } else if (ua.indexOf('Edg') > -1) {
    browserName = 'Edge';
    const match = ua.match(/Edg\/(\d+)/);
    browserVersion = match ? match[1] : 'unknown';
  }

  // OS detection
  if (ua.indexOf('Windows') > -1) {
    os = 'Windows';
  } else if (ua.indexOf('Mac') > -1) {
    os = 'macOS';
  } else if (ua.indexOf('Linux') > -1) {
    os = 'Linux';
  } else if (ua.indexOf('Android') > -1) {
    os = 'Android';
  } else if (ua.indexOf('iOS') > -1 || ua.indexOf('iPhone') > -1 || ua.indexOf('iPad') > -1) {
    os = 'iOS';
  }

  return { name: browserName, version: browserVersion, os };
}



