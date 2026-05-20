import { useState, useEffect } from 'react';
import { getDeviceInfo, getNetworkInfo, getBatteryInfo, isTablet, isPhone } from '@/utils/device-info';

interface DeviceInfo {
  brand: string | null;
  manufacturer: string | null;
  modelName: string | null;
  osName: string | null;
  osVersion: string | null;
  deviceType: number | null;
  isDevice: boolean;
}

interface NetworkInfo {
  isConnected: boolean | null;
  isInternetReachable: boolean | null;
  type: number;
}

interface BatteryInfo {
  batteryLevel: number;
  batteryState: number;
  isLowPowerModeEnabled: boolean;
}

export function useDeviceInfo() {
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo | null>(null);
  const [networkInfo, setNetworkInfo] = useState<NetworkInfo | null>(null);
  const [batteryInfo, setBatteryInfo] = useState<BatteryInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadInfo() {
      try {
        const [device, network, battery] = await Promise.all([
          getDeviceInfo(),
          getNetworkInfo(),
          getBatteryInfo(),
        ]);

        setDeviceInfo(device);
        setNetworkInfo(network);
        setBatteryInfo(battery);
      } catch (error) {
        console.error('Error loading device info:', error);
      } finally {
        setLoading(false);
      }
    }

    loadInfo();
  }, []);

  return {
    deviceInfo,
    networkInfo,
    batteryInfo,
    loading,
    isTablet: isTablet(),
    isPhone: isPhone(),
  };
}

