import { Platform, Dimensions } from 'react-native';
import * as Device from 'expo-device';

export interface DeviceInfo {
  platform: 'ios' | 'android' | 'web';
  isIOS: boolean;
  isAndroid: boolean;
  isWeb: boolean;
  isTablet: boolean;
  isPhone: boolean;
  deviceName?: string;
  osVersion?: string;
  screenWidth: number;
  screenHeight: number;
}

export const getDeviceInfo = (): DeviceInfo => {
  const { width, height } = Dimensions.get('window');
  const isTablet = width >= 768;

  return {
    platform: Platform.OS as 'ios' | 'android' | 'web',
    isIOS: Platform.OS === 'ios',
    isAndroid: Platform.OS === 'android',
    isWeb: Platform.OS === 'web',
    isTablet,
    isPhone: !isTablet,
    deviceName: Device.deviceName || undefined,
    osVersion: Device.osVersion || undefined,
    screenWidth: width,
    screenHeight: height,
  };
};

export const isTablet = (): boolean => {
  const { width } = Dimensions.get('window');
  return width >= 768;
};

export const isPhone = (): boolean => {
  return !isTablet();
};

export const getScreenDimensions = () => {
  return Dimensions.get('window');
};

export const getScreenSize = (): 'small' | 'medium' | 'large' => {
  const { width } = Dimensions.get('window');
  if (width < 375) return 'small';
  if (width < 768) return 'medium';
  return 'large';
};

