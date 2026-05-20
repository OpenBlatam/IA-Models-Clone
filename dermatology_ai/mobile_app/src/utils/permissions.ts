import { Alert, Linking, Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Camera } from 'expo-camera';

/**
 * Request camera permissions
 */
export const requestCameraPermission = async (): Promise<boolean> => {
  try {
    const { status } = await Camera.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(
        'Permiso de Cámara Requerido',
        'Necesitamos acceso a tu cámara para analizar tu piel. Por favor, habilita el permiso en la configuración.',
        [
          { text: 'Cancelar', style: 'cancel' },
          {
            text: 'Abrir Configuración',
            onPress: () => Linking.openSettings(),
          },
        ]
      );
      return false;
    }
    return true;
  } catch (error) {
    console.error('Error requesting camera permission:', error);
    return false;
  }
};

/**
 * Request media library permissions
 */
export const requestMediaLibraryPermission = async (): Promise<boolean> => {
  try {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(
        'Permiso de Galería Requerido',
        'Necesitamos acceso a tu galería para seleccionar imágenes. Por favor, habilita el permiso en la configuración.',
        [
          { text: 'Cancelar', style: 'cancel' },
          {
            text: 'Abrir Configuración',
            onPress: () => Linking.openSettings(),
          },
        ]
      );
      return false;
    }
    return true;
  } catch (error) {
    console.error('Error requesting media library permission:', error);
    return false;
  }
};

/**
 * Check if permissions are granted
 */
export const checkPermissions = async () => {
  const cameraStatus = await Camera.getCameraPermissionsAsync();
  const mediaStatus = await ImagePicker.getMediaLibraryPermissionsAsync();

  return {
    camera: cameraStatus.granted,
    mediaLibrary: mediaStatus.granted,
  };
};

