import { useState, useEffect } from 'react';
import * as Battery from 'expo-battery';

interface BatteryState {
  batteryLevel: number | null;
  isCharging: boolean | null;
  isLowPowerMode: boolean | null;
}

/**
 * Hook to monitor battery status
 * Provides battery level and charging state
 */
export function useBattery(): BatteryState {
  const [batteryState, setBatteryState] = useState<BatteryState>({
    batteryLevel: null,
    isCharging: null,
    isLowPowerMode: null,
  });

  useEffect(() => {
    let isMounted = true;

    async function updateBatteryState() {
      try {
        const [level, charging, lowPowerMode] = await Promise.all([
          Battery.getBatteryLevelAsync(),
          Battery.isBatteryChargingAsync(),
          Battery.isLowPowerModeEnabledAsync(),
        ]);

        if (isMounted) {
          setBatteryState({
            batteryLevel: level,
            isCharging: charging,
            isLowPowerMode: lowPowerMode,
          });
        }
      } catch (error) {
        console.error('Battery state error:', error);
      }
    }

    updateBatteryState();

    const subscriptions = [
      Battery.addBatteryLevelListener(({ batteryLevel }) => {
        if (isMounted) {
          setBatteryState((prev) => ({ ...prev, batteryLevel }));
        }
      }),
      Battery.addBatteryStateListener(({ batteryState: state }) => {
        if (isMounted) {
          setBatteryState((prev) => ({
            ...prev,
            isCharging: state === Battery.BatteryState.CHARGING,
          }));
        }
      }),
    ];

    return () => {
      isMounted = false;
      subscriptions.forEach((sub) => sub.remove());
    };
  }, []);

  return batteryState;
}

