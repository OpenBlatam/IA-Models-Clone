import { useState, useEffect } from 'react';

interface BatteryState {
  charging: boolean;
  chargingTime: number | null;
  dischargingTime: number | null;
  level: number | null;
  supported: boolean;
}

export const useBattery = (): BatteryState => {
  const [state, setState] = useState<BatteryState>({
    charging: false,
    chargingTime: null,
    dischargingTime: null,
    level: null,
    supported: false,
  });

  useEffect(() => {
    if (typeof navigator === 'undefined' || !('getBattery' in navigator)) {
      return;
    }

    const updateBattery = (battery: BatteryManager): void => {
      setState({
        charging: battery.charging,
        chargingTime: battery.chargingTime,
        dischargingTime: battery.dischargingTime,
        level: battery.level,
        supported: true,
      });
    };

    (navigator as any).getBattery().then((battery: BatteryManager) => {
      updateBattery(battery);

      battery.addEventListener('chargingchange', () => updateBattery(battery));
      battery.addEventListener('chargingtimechange', () => updateBattery(battery));
      battery.addEventListener('dischargingtimechange', () => updateBattery(battery));
      battery.addEventListener('levelchange', () => updateBattery(battery));
    });

    return () => {
      // Cleanup handled by browser
    };
  }, []);

  return state;
};



