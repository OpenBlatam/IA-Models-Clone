/**
 * Permission utilities
 */

import * as MediaLibrary from 'expo-media-library';
import * as ImagePicker from 'expo-image-picker';
import * as Camera from 'expo-camera';
import * as Notifications from 'expo-notifications';
import { Platform, Alert, Linking } from 'react-native';

export interface PermissionStatus {
  granted: boolean;
  canAskAgain: boolean;
  status: string;
}

export async function requestMediaLibraryPermission(): Promise<PermissionStatus> {
  try {
    const { status, canAskAgain } = await MediaLibrary.requestPermissionsAsync();
    return {
      granted: status === 'granted',
      canAskAgain: canAskAgain ?? true,
      status,
    };
  } catch (error) {
    console.error('Error requesting media library permission:', error);
    return {
      granted: false,
      canAskAgain: false,
      status: 'undetermined',
    };
  }
}

export async function requestCameraPermission(): Promise<PermissionStatus> {
  try {
    const { status, canAskAgain } = await Camera.requestCameraPermissionsAsync();
    return {
      granted: status === 'granted',
      canAskAgain: canAskAgain ?? true,
      status,
    };
  } catch (error) {
    console.error('Error requesting camera permission:', error);
    return {
      granted: false,
      canAskAgain: false,
      status: 'undetermined',
    };
  }
}

export async function requestImagePickerPermission(): Promise<PermissionStatus> {
  try {
    const { status, canAskAgain } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    return {
      granted: status === 'granted',
      canAskAgain: canAskAgain ?? true,
      status,
    };
  } catch (error) {
    console.error('Error requesting image picker permission:', error);
    return {
      granted: false,
      canAskAgain: false,
      status: 'undetermined',
    };
  }
}

export async function requestNotificationPermission(): Promise<PermissionStatus> {
  try {
    const { status, canAskAgain } = await Notifications.requestPermissionsAsync();
    return {
      granted: status === 'granted',
      canAskAgain: canAskAgain ?? true,
      status,
    };
  } catch (error) {
    console.error('Error requesting notification permission:', error);
    return {
      granted: false,
      canAskAgain: false,
      status: 'undetermined',
    };
  }
}

export async function checkMediaLibraryPermission(): Promise<boolean> {
  try {
    const { status } = await MediaLibrary.getPermissionsAsync();
    return status === 'granted';
  } catch {
    return false;
  }
}

export async function checkCameraPermission(): Promise<boolean> {
  try {
    const { status } = await Camera.getCameraPermissionsAsync();
    return status === 'granted';
  } catch {
    return false;
  }
}

export async function checkNotificationPermission(): Promise<boolean> {
  try {
    const { status } = await Notifications.getPermissionsAsync();
    return status === 'granted';
  } catch {
    return false;
  }
}

export function openSettings(): void {
  if (Platform.OS === 'ios') {
    Linking.openURL('app-settings:');
  } else {
    Linking.openSettings();
  }
}

export function showPermissionAlert(
  title: string,
  message: string,
  onPressSettings?: () => void
): void {
  Alert.alert(
    title,
    message,
    [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Open Settings',
        onPress: onPressSettings || openSettings,
      },
    ]
  );
}

export async function requestMultiplePermissions(
  permissions: Array<() => Promise<PermissionStatus>>
): Promise<Record<string, PermissionStatus>> {
  const results: Record<string, PermissionStatus> = {};

  for (let i = 0; i < permissions.length; i++) {
    const result = await permissions[i]();
    results[`permission_${i}`] = result;
  }

  return results;
}


