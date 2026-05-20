import { useState, useEffect, useCallback } from 'react';
import * as ImagePicker from 'expo-image-picker';
import * as Location from 'expo-location';
import * as Camera from 'expo-camera';

export type PermissionType = 'camera' | 'mediaLibrary' | 'location';

export interface PermissionStatus {
  granted: boolean;
  canAskAgain: boolean;
  status: 'granted' | 'denied' | 'undetermined';
}

export function usePermissions(permissionType: PermissionType) {
  const [permissionStatus, setPermissionStatus] = useState<PermissionStatus>({
    granted: false,
    canAskAgain: true,
    status: 'undetermined',
  });
  const [isLoading, setIsLoading] = useState(true);

  const checkPermission = useCallback(async () => {
    try {
      let status;
      let canAskAgain = true;

      switch (permissionType) {
        case 'camera':
          const cameraStatus = await Camera.getCameraPermissionsAsync();
          status = cameraStatus.status;
          canAskAgain = cameraStatus.canAskAgain;
          break;
        case 'mediaLibrary':
          const mediaStatus = await ImagePicker.getMediaLibraryPermissionsAsync();
          status = mediaStatus.status;
          canAskAgain = mediaStatus.canAskAgain;
          break;
        case 'location':
          const locationStatus = await Location.getForegroundPermissionsAsync();
          status = locationStatus.status;
          canAskAgain = locationStatus.canAskAgain;
          break;
      }

      setPermissionStatus({
        granted: status === 'granted',
        canAskAgain,
        status: status as 'granted' | 'denied' | 'undetermined',
      });
    } catch (error) {
      console.error('Error checking permission:', error);
    } finally {
      setIsLoading(false);
    }
  }, [permissionType]);

  const requestPermission = useCallback(async () => {
    try {
      setIsLoading(true);
      let status;
      let canAskAgain = true;

      switch (permissionType) {
        case 'camera':
          const cameraResult = await Camera.requestCameraPermissionsAsync();
          status = cameraResult.status;
          canAskAgain = cameraResult.canAskAgain;
          break;
        case 'mediaLibrary':
          const mediaResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
          status = mediaResult.status;
          canAskAgain = mediaResult.canAskAgain;
          break;
        case 'location':
          const locationResult = await Location.requestForegroundPermissionsAsync();
          status = locationResult.status;
          canAskAgain = locationResult.canAskAgain;
          break;
      }

      setPermissionStatus({
        granted: status === 'granted',
        canAskAgain,
        status: status as 'granted' | 'denied' | 'undetermined',
      });
      return status === 'granted';
    } catch (error) {
      console.error('Error requesting permission:', error);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [permissionType]);

  useEffect(() => {
    checkPermission();
  }, [checkPermission]);

  return {
    ...permissionStatus,
    isLoading,
    requestPermission,
    checkPermission,
  };
}

