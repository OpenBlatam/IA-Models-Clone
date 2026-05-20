import { useState, useRef, useEffect } from 'react';
import { Camera } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import { Alert } from 'react-native';

interface UseCameraReturn {
  hasPermission: boolean | null;
  cameraType: typeof Camera.Constants.Type.back | typeof Camera.Constants.Type.front;
  isRecording: boolean;
  isProcessing: boolean;
  cameraRef: React.RefObject<Camera>;
  toggleCameraType: () => void;
  takePicture: () => Promise<string | null>;
  pickImage: () => Promise<string | null>;
  recordVideo: () => Promise<string | null>;
  requestPermissions: () => Promise<void>;
}

export const useCamera = (): UseCameraReturn => {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [cameraType, setCameraType] = useState<typeof Camera.Constants.Type.back | typeof Camera.Constants.Type.front>(
    Camera.Constants.Type.back
  );
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const cameraRef = useRef<Camera>(null);

  useEffect(() => {
    requestPermissions();
  }, []);

  const requestPermissions = async () => {
    try {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    } catch (error) {
      console.error('Error requesting camera permissions:', error);
      setHasPermission(false);
    }
  };

  const toggleCameraType = () => {
    setCameraType(
      cameraType === Camera.Constants.Type.back
        ? Camera.Constants.Type.front
        : Camera.Constants.Type.back
    );
  };

  const takePicture = async (): Promise<string | null> => {
    if (!cameraRef.current) return null;

    try {
      setIsProcessing(true);
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        base64: false,
      });
      return photo.uri;
    } catch (error) {
      Alert.alert('Error', 'No se pudo tomar la foto');
      console.error(error);
      return null;
    } finally {
      setIsProcessing(false);
    }
  };

  const pickImage = async (): Promise<string | null> => {
    try {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert(
          'Permisos necesarios',
          'Necesitamos acceso a tu galería para seleccionar imágenes'
        );
        return null;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        return result.assets[0].uri;
      }
      return null;
    } catch (error) {
      Alert.alert('Error', 'No se pudo seleccionar la imagen');
      console.error(error);
      return null;
    }
  };

  const recordVideo = async (): Promise<string | null> => {
    if (!cameraRef.current) return null;

    try {
      if (!isRecording) {
        setIsRecording(true);
        const video = await cameraRef.current.recordAsync({
          quality: Camera.Constants.VideoQuality['720p'],
        });
        setIsRecording(false);
        return video.uri;
      } else {
        cameraRef.current.stopRecording();
        setIsRecording(false);
        return null;
      }
    } catch (error) {
      Alert.alert('Error', 'No se pudo grabar el video');
      console.error(error);
      setIsRecording(false);
      return null;
    }
  };

  return {
    hasPermission,
    cameraType,
    isRecording,
    isProcessing,
    cameraRef,
    toggleCameraType,
    takePicture,
    pickImage,
    recordVideo,
    requestPermissions,
  };
};

