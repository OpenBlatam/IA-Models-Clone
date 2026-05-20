import { Platform, Dimensions, PixelRatio } from 'react-native';
import * as Device from 'expo-device';

/**
 * Device utilities
 */

/**
 * Get device info
 */
export const getDeviceInfo = () => {
  return {
    platform: Platform.OS,
    version: Platform.Version,
    isIOS: Platform.OS === 'ios',
    isAndroid: Platform.OS === 'android',
    deviceName: Device.deviceName,
    modelName: Device.modelName,
    osVersion: Device.osVersion,
  };
};

/**
 * Get screen dimensions
 */
export const getScreenDimensions = () => {
  const { width, height } = Dimensions.get('window');
  return {
    width,
    height,
    scale: PixelRatio.get(),
    fontScale: PixelRatio.getFontScale(),
  };
};

/**
 * Check if device is tablet
 */
export const isTablet = (): boolean => {
  const { width, height } = Dimensions.get('window');
  const aspectRatio = height / width;
  return (
    (Platform.OS === 'ios' && aspectRatio < 1.6) ||
    (Platform.OS === 'android' && aspectRatio < 1.6)
  );
};

/**
 * Check if device is phone
 */
export const isPhone = (): boolean => {
  return !isTablet();
};

/**
 * Get responsive value
 */
export const getResponsiveValue = <T,>(
  phoneValue: T,
  tabletValue: T
): T => {
  return isTablet() ? tabletValue : phoneValue;
};

/**
 * Normalize size for different screen densities
 */
export const normalize = (size: number): number => {
  const scale = Dimensions.get('window').width / 320;
  const newSize = size * scale;
  return Math.round(PixelRatio.roundToNearestPixel(newSize));
};

/**
 * Check if device has notch
 */
export const hasNotch = async (): Promise<boolean> => {
  try {
    return await Device.hasNotchAsync();
  } catch {
    return false;
  }
};

/**
 * Get device type
 */
export const getDeviceType = (): Device.DeviceType => {
  return Device.deviceType || Device.DeviceType.UNKNOWN;
};

