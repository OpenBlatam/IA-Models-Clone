/**
 * Permissions Utilities
 * ====================
 * Utilities for handling permissions
 */

import * as ImagePicker from 'expo-image-picker';
import { Alert } from 'react-native';
import { useTranslation } from '@/hooks/use-translation';

export async function requestCameraPermission(): Promise<boolean> {
  const { status } = await ImagePicker.requestCameraPermissionsAsync();
  if (status !== 'granted') {
    Alert.alert(
      'Permission Required',
      'Camera permission is required to take photos. Please enable it in settings.'
    );
    return false;
  }
  return true;
}

export async function requestMediaLibraryPermission(): Promise<boolean> {
  const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
  if (status !== 'granted') {
    Alert.alert(
      'Permission Required',
      'Media library permission is required to select photos. Please enable it in settings.'
    );
    return false;
  }
  return true;
}

export async function checkCameraPermission(): Promise<boolean> {
  const { status } = await ImagePicker.getCameraPermissionsAsync();
  return status === 'granted';
}

export async function checkMediaLibraryPermission(): Promise<boolean> {
  const { status } = await ImagePicker.getMediaLibraryPermissionsAsync();
  return status === 'granted';
}



