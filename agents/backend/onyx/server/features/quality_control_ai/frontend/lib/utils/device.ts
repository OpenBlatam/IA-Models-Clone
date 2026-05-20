export const isMobile = (): boolean => {
  if (typeof window === 'undefined') return false;
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent
  );
};

export const isTablet = (): boolean => {
  if (typeof window === 'undefined') return false;
  return /iPad|Android/i.test(navigator.userAgent) && !isMobile();
};

export const isDesktop = (): boolean => {
  return !isMobile() && !isTablet();
};

export const isIOS = (): boolean => {
  if (typeof window === 'undefined') return false;
  return /iPad|iPhone|iPod/.test(navigator.userAgent);
};

export const isAndroid = (): boolean => {
  if (typeof window === 'undefined') return false;
  return /Android/.test(navigator.userAgent);
};

export const isWindows = (): boolean => {
  if (typeof window === 'undefined') return false;
  return /Windows/.test(navigator.userAgent);
};

export const isMac = (): boolean => {
  if (typeof window === 'undefined') return false;
  return /Mac/.test(navigator.userAgent);
};

export const isLinux = (): boolean => {
  if (typeof window === 'undefined') return false;
  return /Linux/.test(navigator.userAgent);
};

export const getDeviceType = (): 'mobile' | 'tablet' | 'desktop' => {
  if (isMobile()) return 'mobile';
  if (isTablet()) return 'tablet';
  return 'desktop';
};

export const getOS = (): 'ios' | 'android' | 'windows' | 'mac' | 'linux' | 'unknown' => {
  if (isIOS()) return 'ios';
  if (isAndroid()) return 'android';
  if (isWindows()) return 'windows';
  if (isMac()) return 'mac';
  if (isLinux()) return 'linux';
  return 'unknown';
};

export const isTouchDevice = (): boolean => {
  if (typeof window === 'undefined') return false;
  return 'ontouchstart' in window || navigator.maxTouchPoints > 0;
};

export const isStandalone = (): boolean => {
  if (typeof window === 'undefined') return false;
  return (
    (window.navigator as Navigator & { standalone?: boolean }).standalone === true ||
    window.matchMedia('(display-mode: standalone)').matches
  );
};

