export const isMobile = (): boolean => {
  if (typeof window === 'undefined') {
    return false;
  }
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
};

export const isIOS = (): boolean => {
  if (typeof window === 'undefined') {
    return false;
  }
  return /iPad|iPhone|iPod/.test(navigator.userAgent);
};

export const isAndroid = (): boolean => {
  if (typeof window === 'undefined') {
    return false;
  }
  return /Android/.test(navigator.userAgent);
};

export const isDesktop = (): boolean => {
  return !isMobile();
};

export const isTouchDevice = (): boolean => {
  if (typeof window === 'undefined') {
    return false;
  }
  return 'ontouchstart' in window || navigator.maxTouchPoints > 0;
};

export const getDeviceType = (): 'mobile' | 'tablet' | 'desktop' => {
  if (typeof window === 'undefined') {
    return 'desktop';
  }

  const width = window.innerWidth;
  if (width < 768) {
    return 'mobile';
  }
  if (width < 1024) {
    return 'tablet';
  }
  return 'desktop';
};

export const getBrowserInfo = (): {
  name: string;
  version: string;
  os: string;
} => {
  if (typeof window === 'undefined') {
    return { name: 'Unknown', version: '0', os: 'Unknown' };
  }

  const ua = navigator.userAgent;
  let browserName = 'Unknown';
  let browserVersion = '0';
  let os = 'Unknown';

  if (ua.indexOf('Chrome') > -1) {
    browserName = 'Chrome';
    browserVersion = ua.match(/Chrome\/(\d+)/)?.[1] || '0';
  } else if (ua.indexOf('Firefox') > -1) {
    browserName = 'Firefox';
    browserVersion = ua.match(/Firefox\/(\d+)/)?.[1] || '0';
  } else if (ua.indexOf('Safari') > -1) {
    browserName = 'Safari';
    browserVersion = ua.match(/Version\/(\d+)/)?.[1] || '0';
  } else if (ua.indexOf('Edge') > -1) {
    browserName = 'Edge';
    browserVersion = ua.match(/Edge\/(\d+)/)?.[1] || '0';
  }

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
};



