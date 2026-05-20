/**
 * Device utility functions.
 * Provides helper functions for device detection and capabilities.
 */

/**
 * Checks if running on mobile device.
 * @returns True if mobile device
 */
export function isMobile(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }

  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent
  );
}

/**
 * Checks if running on tablet device.
 * @returns True if tablet device
 */
export function isTablet(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }

  return /iPad|Android/i.test(navigator.userAgent) && !isMobile();
}

/**
 * Checks if running on desktop device.
 * @returns True if desktop device
 */
export function isDesktop(): boolean {
  return !isMobile() && !isTablet();
}

/**
 * Gets device type.
 * @returns Device type
 */
export function getDeviceType(): 'mobile' | 'tablet' | 'desktop' {
  if (isMobile()) return 'mobile';
  if (isTablet()) return 'tablet';
  return 'desktop';
}

/**
 * Checks if device supports touch.
 * @returns True if touch supported
 */
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

/**
 * Gets user agent string.
 * @returns User agent
 */
export function getUserAgent(): string {
  if (typeof navigator === 'undefined') {
    return '';
  }
  return navigator.userAgent;
}

/**
 * Gets platform string.
 * @returns Platform
 */
export function getPlatform(): string {
  if (typeof navigator === 'undefined') {
    return '';
  }
  return navigator.platform;
}

/**
 * Checks if running on iOS.
 * @returns True if iOS
 */
export function isIOS(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }

  return /iPad|iPhone|iPod/.test(navigator.userAgent);
}

/**
 * Checks if running on Android.
 * @returns True if Android
 */
export function isAndroid(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }

  return /Android/.test(navigator.userAgent);
}

/**
 * Checks if running on Windows.
 * @returns True if Windows
 */
export function isWindows(): boolean {
  if (typeof navigator === 'undefined') {
    return false;
  }

  return /Win/.test(navigator.platform);
}

/**
 * Checks if running on Mac.
 * @returns True if Mac
 */
export function isMac(): boolean {
  if (typeof navigator === 'undefined') {
    return false;
  }

  return /Mac/.test(navigator.platform);
}

/**
 * Checks if running on Linux.
 * @returns True if Linux
 */
export function isLinux(): boolean {
  if (typeof navigator === 'undefined') {
    return false;
  }

  return /Linux/.test(navigator.platform);
}

/**
 * Gets browser name.
 * @returns Browser name
 */
export function getBrowser(): string {
  if (typeof navigator === 'undefined') {
    return 'unknown';
  }

  const ua = navigator.userAgent;

  if (ua.includes('Chrome') && !ua.includes('Edg')) {
    return 'chrome';
  }
  if (ua.includes('Firefox')) {
    return 'firefox';
  }
  if (ua.includes('Safari') && !ua.includes('Chrome')) {
    return 'safari';
  }
  if (ua.includes('Edg')) {
    return 'edge';
  }
  if (ua.includes('Opera') || ua.includes('OPR')) {
    return 'opera';
  }

  return 'unknown';
}

