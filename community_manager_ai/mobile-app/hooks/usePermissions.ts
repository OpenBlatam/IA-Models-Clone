import { useState, useEffect } from 'react';
import * as ImagePicker from 'expo-image-picker';
import { Alert, Platform } from 'react-native';

interface PermissionsState {
  camera: boolean | null;
  mediaLibrary: boolean | null;
  mediaLibraryWrite: boolean | null;
}

export function usePermissions() {
  const [permissions, setPermissions] = useState<PermissionsState>({
    camera: null,
    mediaLibrary: null,
    mediaLibraryWrite: null,
  });

  const [loading, setLoading] = useState(false);

  useEffect(() => {
    checkPermissions();
  }, []);

  async function checkPermissions() {
    setLoading(true);
    try {
      const [camera, mediaLibrary] = await Promise.all([
        ImagePicker.getCameraPermissionsAsync(),
        ImagePicker.getMediaLibraryPermissionsAsync(),
      ]);

      setPermissions({
        camera: camera.granted,
        mediaLibrary: mediaLibrary.granted,
        mediaLibraryWrite: mediaLibrary.granted, // Simplified for now
      });
    } catch (error) {
      console.error('Error checking permissions:', error);
    } finally {
      setLoading(false);
    }
  }

  async function requestCameraPermission(): Promise<boolean> {
    try {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      const granted = status === 'granted';
      setPermissions((prev) => ({ ...prev, camera: granted }));
      
      if (!granted) {
        Alert.alert(
          'Permission Required',
          'Camera permission is required to take photos. Please enable it in settings.',
          [{ text: 'OK' }]
        );
      }
      
      return granted;
    } catch (error) {
      console.error('Error requesting camera permission:', error);
      return false;
    }
  }

  async function requestMediaLibraryPermission(): Promise<boolean> {
    try {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      const granted = status === 'granted';
      setPermissions((prev) => ({ ...prev, mediaLibrary: granted }));
      
      if (!granted) {
        Alert.alert(
          'Permission Required',
          'Media library permission is required to select images. Please enable it in settings.',
          [{ text: 'OK' }]
        );
      }
      
      return granted;
    } catch (error) {
      console.error('Error requesting media library permission:', error);
      return false;
    }
  }

  return {
    permissions,
    loading,
    requestCameraPermission,
    requestMediaLibraryPermission,
    checkPermissions,
  };
}

