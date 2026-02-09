import * as Device from 'expo-device';
import * as Network from 'expo-network';
import * as Battery from 'expo-battery';

// Pure functions for device information

export async function getDeviceInfo() {
  return {
    brand: Device.brand,
    manufacturer: Device.manufacturer,
    modelName: Device.modelName,
    osName: Device.osName,
    osVersion: Device.osVersion,
    deviceType: Device.deviceType,
    isDevice: Device.isDevice,
  };
}

export async function getNetworkInfo() {
  const networkState = await Network.getNetworkStateAsync();
  return {
    isConnected: networkState.isConnected,
    isInternetReachable: networkState.isInternetReachable,
    type: networkState.type,
  };
}

export async function getBatteryInfo() {
  const batteryLevel = await Battery.getBatteryLevelAsync();
  const batteryState = await Battery.getBatteryStateAsync();
  const isLowPowerModeEnabled = await Battery.isLowPowerModeEnabledAsync();

  return {
    batteryLevel,
    batteryState,
    isLowPowerModeEnabled,
  };
}

export function isTablet(): boolean {
  return Device.deviceType === Device.DeviceType.TABLET;
}

export function isPhone(): boolean {
  return Device.deviceType === Device.DeviceType.PHONE;
}

