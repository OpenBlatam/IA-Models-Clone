import { useEffect, useState } from 'react';
import * as Updates from 'expo-updates';
import { Alert, Platform } from 'react-native';

interface UpdateInfo {
  isUpdateAvailable: boolean;
  isUpdatePending: boolean;
  isChecking: boolean;
  isDownloading: boolean;
  lastError: Error | null;
}

export function useUpdates() {
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo>({
    isUpdateAvailable: false,
    isUpdatePending: false,
    isChecking: false,
    isDownloading: false,
    lastError: null,
  });

  useEffect(() => {
    checkForUpdates();
  }, []);

  async function checkForUpdates() {
    if (!Updates.isEnabled) {
      return;
    }

    setUpdateInfo((prev) => ({ ...prev, isChecking: true }));

    try {
      const update = await Updates.checkForUpdateAsync();

      if (update.isAvailable) {
        setUpdateInfo((prev) => ({
          ...prev,
          isUpdateAvailable: true,
          isChecking: false,
        }));

        // Automatically download update
        await downloadUpdate();
      } else {
        setUpdateInfo((prev) => ({
          ...prev,
          isChecking: false,
        }));
      }
    } catch (error) {
      setUpdateInfo((prev) => ({
        ...prev,
        isChecking: false,
        lastError: error as Error,
      }));
    }
  }

  async function downloadUpdate() {
    if (!Updates.isEnabled) {
      return;
    }

    setUpdateInfo((prev) => ({ ...prev, isDownloading: true }));

    try {
      const result = await Updates.fetchUpdateAsync();

      if (result.isNew) {
        setUpdateInfo((prev) => ({
          ...prev,
          isUpdatePending: true,
          isDownloading: false,
        }));

        Alert.alert(
          'Update Available',
          'A new version is available. Restart the app to apply the update.',
          [
            { text: 'Later', style: 'cancel' },
            {
              text: 'Restart',
              onPress: async () => {
                await Updates.reloadAsync();
              },
            },
          ]
        );
      }
    } catch (error) {
      setUpdateInfo((prev) => ({
        ...prev,
        isDownloading: false,
        lastError: error as Error,
      }));
    }
  }

  async function reloadApp() {
    if (Updates.isEnabled) {
      await Updates.reloadAsync();
    }
  }

  return {
    ...updateInfo,
    checkForUpdates,
    downloadUpdate,
    reloadApp,
  };
}


