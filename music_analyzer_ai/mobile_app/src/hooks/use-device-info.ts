import { useState, useEffect } from 'react';
import { Platform, Dimensions } from 'react-native';
import * as Device from 'expo-device';
import Constants from 'expo-constants';

interface DeviceInfo {
  platform: 'ios' | 'android' | 'web';
  osVersion: string | null;
  deviceName: string | null;
  deviceType: Device.DeviceType | null;
  brand: string | null;
  modelName: string | null;
  isDevice: boolean;
  isTablet: boolean;
  isPhone: boolean;
  screenWidth: number;
  screenHeight: number;
  screenScale: number;
  appVersion: string | null;
  buildNumber: string | null;
}

/**
 * Hook to get comprehensive device information
 */
export function useDeviceInfo(): DeviceInfo {
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo>({
    platform: Platform.OS as 'ios' | 'android' | 'web',
    osVersion: null,
    deviceName: null,
    deviceType: null,
    brand: null,
    modelName: null,
    isDevice: Device.isDevice,
    isTablet: false,
    isPhone: false,
    screenWidth: Dimensions.get('window').width,
    screenHeight: Dimensions.get('window').height,
    screenScale: Dimensions.get('window').scale,
    appVersion: Constants.expoConfig?.version || null,
    buildNumber: Constants.expoConfig?.ios?.buildNumber || Constants.expoConfig?.android?.versionCode?.toString() || null,
  });

  useEffect(() => {
    async function loadDeviceInfo() {
      try {
        const [deviceType, brand, modelName] = await Promise.all([
          Device.getDeviceTypeAsync(),
          Device.brand,
          Device.modelName,
        ]);

        const osVersion = Platform.Version.toString();
        const screen = Dimensions.get('window');
        const isTablet = deviceType === Device.DeviceType.TABLET;
        const isPhone = deviceType === Device.DeviceType.PHONE;

        setDeviceInfo((prev) => ({
          ...prev,
          osVersion,
          deviceType,
          brand,
          modelName,
          isTablet,
          isPhone,
          screenWidth: screen.width,
          screenHeight: screen.height,
          screenScale: screen.scale,
        }));
      } catch (error) {
        console.error('Error loading device info:', error);
      }
    }

    loadDeviceInfo();

    const subscription = Dimensions.addEventListener('change', ({ window }) => {
      setDeviceInfo((prev) => ({
        ...prev,
        screenWidth: window.width,
        screenHeight: window.height,
        screenScale: window.scale,
      }));
    });

    return () => subscription?.remove();
  }, []);

  return deviceInfo;
}

