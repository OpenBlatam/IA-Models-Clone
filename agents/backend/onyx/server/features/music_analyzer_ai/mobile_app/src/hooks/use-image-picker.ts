import { useState, useCallback } from 'react';
import * as ImagePicker from 'expo-image-picker';
import { Alert, Platform } from 'react-native';

interface ImagePickerOptions {
  allowsEditing?: boolean;
  aspect?: [number, number];
  quality?: number;
  mediaTypes?: ImagePicker.MediaTypeOptions;
}

interface ImageResult {
  uri: string;
  width: number;
  height: number;
  type: string;
}

/**
 * Hook for image picking functionality
 * Handles permissions and image selection
 */
export function useImagePicker() {
  const [image, setImage] = useState<ImageResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const requestPermissions = useCallback(async (): Promise<boolean> => {
    if (Platform.OS !== 'web') {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert(
          'Permission Required',
          'We need camera roll permissions to select images.'
        );
        return false;
      }
    }
    return true;
  }, []);

  const pickImage = useCallback(
    async (options: ImagePickerOptions = {}) => {
      const hasPermission = await requestPermissions();
      if (!hasPermission) {
        return null;
      }

      setIsLoading(true);
      try {
        const result = await ImagePicker.launchImageLibraryAsync({
          mediaTypes: options.mediaTypes || ImagePicker.MediaTypeOptions.Images,
          allowsEditing: options.allowsEditing ?? true,
          aspect: options.aspect,
          quality: options.quality ?? 1,
        });

        if (!result.canceled && result.assets && result.assets.length > 0) {
          const asset = result.assets[0];
          const imageResult: ImageResult = {
            uri: asset.uri,
            width: asset.width,
            height: asset.height,
            type: asset.type || 'image',
          };
          setImage(imageResult);
          return imageResult;
        }
        return null;
      } catch (error) {
        console.error('Image picker error:', error);
        Alert.alert('Error', 'Failed to pick image');
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [requestPermissions]
  );

  const takePhoto = useCallback(
    async (options: ImagePickerOptions = {}) => {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert(
          'Permission Required',
          'We need camera permissions to take photos.'
        );
        return null;
      }

      setIsLoading(true);
      try {
        const result = await ImagePicker.launchCameraAsync({
          allowsEditing: options.allowsEditing ?? true,
          aspect: options.aspect,
          quality: options.quality ?? 1,
        });

        if (!result.canceled && result.assets && result.assets.length > 0) {
          const asset = result.assets[0];
          const imageResult: ImageResult = {
            uri: asset.uri,
            width: asset.width,
            height: asset.height,
            type: asset.type || 'image',
          };
          setImage(imageResult);
          return imageResult;
        }
        return null;
      } catch (error) {
        console.error('Camera error:', error);
        Alert.alert('Error', 'Failed to take photo');
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const clearImage = useCallback(() => {
    setImage(null);
  }, []);

  return {
    image,
    isLoading,
    pickImage,
    takePhoto,
    clearImage,
  };
}

