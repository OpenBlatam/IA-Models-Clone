import { useState, useEffect } from 'react';
import {
  isMobile,
  isTablet,
  isDesktop,
  isIOS,
  isAndroid,
  isTouchDevice,
  getDeviceType,
  getBrowserInfo,
} from '@/lib/utils/device';

export function useDevice() {
  const [deviceInfo, setDeviceInfo] = useState({
    isMobile: isMobile(),
    isTablet: isTablet(),
    isDesktop: isDesktop(),
    isIOS: isIOS(),
    isAndroid: isAndroid(),
    isTouch: isTouchDevice(),
    deviceType: getDeviceType(),
    browser: getBrowserInfo(),
  });

  useEffect(() => {
    const updateDeviceInfo = () => {
      setDeviceInfo({
        isMobile: isMobile(),
        isTablet: isTablet(),
        isDesktop: isDesktop(),
        isIOS: isIOS(),
        isAndroid: isAndroid(),
        isTouch: isTouchDevice(),
        deviceType: getDeviceType(),
        browser: getBrowserInfo(),
      });
    };

    window.addEventListener('resize', updateDeviceInfo);
    return () => window.removeEventListener('resize', updateDeviceInfo);
  }, []);

  return deviceInfo;
}



