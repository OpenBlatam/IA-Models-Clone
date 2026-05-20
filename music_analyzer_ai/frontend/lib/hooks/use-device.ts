/**
 * Custom hook for device detection.
 * Provides reactive device type and capabilities.
 */

import { useState, useEffect } from 'react';
import {
  isMobile,
  isTablet,
  isDesktop,
  getDeviceType,
  isTouchDevice,
  getBrowser,
  isIOS,
  isAndroid,
} from '../utils/device';

/**
 * Device information interface.
 */
export interface DeviceInfo {
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  deviceType: 'mobile' | 'tablet' | 'desktop';
  isTouch: boolean;
  browser: string;
  isIOS: boolean;
  isAndroid: boolean;
}

/**
 * Custom hook for device detection.
 * Provides reactive device information.
 *
 * @returns Device information
 */
export function useDevice(): DeviceInfo {
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo>(() => ({
    isMobile: isMobile(),
    isTablet: isTablet(),
    isDesktop: isDesktop(),
    deviceType: getDeviceType(),
    isTouch: isTouchDevice(),
    browser: getBrowser(),
    isIOS: isIOS(),
    isAndroid: isAndroid(),
  }));

  useEffect(() => {
    const updateDeviceInfo = () => {
      setDeviceInfo({
        isMobile: isMobile(),
        isTablet: isTablet(),
        isDesktop: isDesktop(),
        deviceType: getDeviceType(),
        isTouch: isTouchDevice(),
        browser: getBrowser(),
        isIOS: isIOS(),
        isAndroid: isAndroid(),
      });
    };

    // Update on resize (orientation change, etc.)
    window.addEventListener('resize', updateDeviceInfo);
    window.addEventListener('orientationchange', updateDeviceInfo);

    return () => {
      window.removeEventListener('resize', updateDeviceInfo);
      window.removeEventListener('orientationchange', updateDeviceInfo);
    };
  }, []);

  return deviceInfo;
}

