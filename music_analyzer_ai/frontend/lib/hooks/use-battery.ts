/**
 * Custom hook for battery status detection.
 * Provides reactive battery status information.
 */

import { useState, useEffect } from 'react';

/**
 * Battery status interface.
 */
export interface BatteryStatus {
  level: number; // 0-1
  charging: boolean;
  chargingTime: number | null;
  dischargingTime: number | null;
  supported: boolean;
}

/**
 * Custom hook for battery status.
 * Tracks device battery level and charging status.
 *
 * @returns Battery status
 */
export function useBattery(): BatteryStatus {
  const [status, setStatus] = useState<BatteryStatus>(() => {
    if (typeof navigator === 'undefined' || !('getBattery' in navigator)) {
      return {
        level: 1,
        charging: false,
        chargingTime: null,
        dischargingTime: null,
        supported: false,
      };
    }

    return {
      level: 1,
      charging: false,
      chargingTime: null,
      dischargingTime: null,
      supported: true,
    };
  });

  useEffect(() => {
    if (typeof navigator === 'undefined' || !('getBattery' in navigator)) {
      return;
    }

    let battery: any = null;

    (navigator as any)
      .getBattery()
      .then((batteryManager: any) => {
        battery = batteryManager;

        const updateStatus = () => {
          setStatus({
            level: battery.level,
            charging: battery.charging,
            chargingTime: battery.chargingTime,
            dischargingTime: battery.dischargingTime,
            supported: true,
          });
        };

        updateStatus();

        battery.addEventListener('chargingchange', updateStatus);
        battery.addEventListener('levelchange', updateStatus);
        battery.addEventListener('chargingtimechange', updateStatus);
        battery.addEventListener('dischargingtimechange', updateStatus);

        return () => {
          if (battery) {
            battery.removeEventListener('chargingchange', updateStatus);
            battery.removeEventListener('levelchange', updateStatus);
            battery.removeEventListener('chargingtimechange', updateStatus);
            battery.removeEventListener('dischargingtimechange', updateStatus);
          }
        };
      })
      .catch(() => {
        setStatus((prev) => ({ ...prev, supported: false }));
      });
  }, []);

  return status;
}

