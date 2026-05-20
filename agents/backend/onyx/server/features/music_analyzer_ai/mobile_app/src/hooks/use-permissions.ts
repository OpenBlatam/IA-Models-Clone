import { useState, useEffect } from 'react';
import * as MediaLibrary from 'expo-media-library';
import * as Notifications from 'expo-notifications';

interface PermissionStatus {
  granted: boolean;
  canAskAgain: boolean;
  status: string;
}

/**
 * Hook to request and check media library permissions
 */
export function useMediaLibraryPermission(): [
  PermissionStatus | null,
  () => Promise<PermissionStatus>
] {
  const [permission, setPermission] = useState<PermissionStatus | null>(null);

  useEffect(() => {
    checkPermission();
  }, []);

  async function checkPermission() {
    const { status, canAskAgain } = await MediaLibrary.getPermissionsAsync();
    setPermission({
      granted: status === 'granted',
      canAskAgain,
      status,
    });
  }

  async function requestPermission(): Promise<PermissionStatus> {
    const { status, canAskAgain } = await MediaLibrary.requestPermissionsAsync();
    const permissionStatus = {
      granted: status === 'granted',
      canAskAgain,
      status,
    };
    setPermission(permissionStatus);
    return permissionStatus;
  }

  return [permission, requestPermission];
}

/**
 * Hook to request and check notification permissions
 */
export function useNotificationPermission(): [
  PermissionStatus | null,
  () => Promise<PermissionStatus>
] {
  const [permission, setPermission] = useState<PermissionStatus | null>(null);

  useEffect(() => {
    checkPermission();
  }, []);

  async function checkPermission() {
    const { status, canAskAgain } = await Notifications.getPermissionsAsync();
    setPermission({
      granted: status === 'granted',
      canAskAgain,
      status,
    });
  }

  async function requestPermission(): Promise<PermissionStatus> {
    const { status, canAskAgain } = await Notifications.requestPermissionsAsync();
    const permissionStatus = {
      granted: status === 'granted',
      canAskAgain,
      status,
    };
    setPermission(permissionStatus);
    return permissionStatus;
  }

  return [permission, requestPermission];
}

