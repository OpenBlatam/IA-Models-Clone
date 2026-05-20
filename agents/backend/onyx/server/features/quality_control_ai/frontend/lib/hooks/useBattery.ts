import { useEffect, useState } from 'react';

interface BatteryStatus {
  charging: boolean;
  chargingTime: number;
  dischargingTime: number;
  level: number;
}

export const useBattery = () => {
  const [battery, setBattery] = useState<BatteryStatus | null>(null);
  const [isSupported, setIsSupported] = useState(false);

  useEffect(() => {
    if (typeof navigator === 'undefined' || !('getBattery' in navigator)) {
      return;
    }

    setIsSupported(true);

    const getBatteryStatus = async (): Promise<void> => {
      try {
        const batteryManager = await (navigator as Navigator & {
          getBattery: () => Promise<BatteryManager>;
        }).getBattery();

        const updateBattery = (): void => {
          setBattery({
            charging: batteryManager.charging,
            chargingTime: batteryManager.chargingTime,
            dischargingTime: batteryManager.dischargingTime,
            level: batteryManager.level,
          });
        };

        updateBattery();

        batteryManager.addEventListener('chargingchange', updateBattery);
        batteryManager.addEventListener('chargingtimechange', updateBattery);
        batteryManager.addEventListener('dischargingtimechange', updateBattery);
        batteryManager.addEventListener('levelchange', updateBattery);

        return () => {
          batteryManager.removeEventListener('chargingchange', updateBattery);
          batteryManager.removeEventListener('chargingtimechange', updateBattery);
          batteryManager.removeEventListener('dischargingtimechange', updateBattery);
          batteryManager.removeEventListener('levelchange', updateBattery);
        };
      } catch {
        setIsSupported(false);
      }
    };

    getBatteryStatus();
  }, []);

  return { battery, isSupported };
};

