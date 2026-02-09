import { useState, useCallback } from 'react';
import * as ImagePicker from 'expo-image-picker';
import { Alert, Platform } from 'react-native';

export interface ImagePickerOptions {
  allowsEditing?: boolean;
  aspect?: [number, number];
  quality?: number;
  mediaTypes?: ImagePicker.MediaTypeOptions;
}

export interface ImageInfo {
  uri: string;
  width: number;
  height: number;
  type?: string;
  fileName?: string;
  fileSize?: number;
}

export function useImagePicker(options: ImagePickerOptions = {}) {
  const [image, setImage] = useState<ImageInfo | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const pickImage = useCallback(
    async (source: 'camera' | 'library' = 'library') => {
      try {
        setIsLoading(true);
        setError(null);

        // Request permissions
        if (source === 'camera') {
          const { status } = await ImagePicker.requestCameraPermissionsAsync();
          if (status !== 'granted') {
            Alert.alert('Permission required', 'Camera permission is required to take photos');
            return;
          }
        } else {
          const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
          if (status !== 'granted') {
            Alert.alert('Permission required', 'Media library permission is required');
            return;
          }
        }

        // Pick image
        let result: ImagePicker.ImagePickerResult;
        if (source === 'camera') {
          result = await ImagePicker.launchCameraAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: options.allowsEditing ?? true,
            aspect: options.aspect,
            quality: options.quality ?? 0.8,
          });
        } else {
          result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: options.mediaTypes ?? ImagePicker.MediaTypeOptions.Images,
            allowsEditing: options.allowsEditing ?? true,
            aspect: options.aspect,
            quality: options.quality ?? 0.8,
          });
        }

        if (!result.canceled && result.assets && result.assets.length > 0) {
          const asset = result.assets[0];
          setImage({
            uri: asset.uri,
            width: asset.width,
            height: asset.height,
            type: asset.type,
            fileName: asset.fileName,
            fileSize: asset.fileSize,
          });
        }
      } catch (err) {
        const error = err instanceof Error ? err : new Error('Failed to pick image');
        setError(error);
        Alert.alert('Error', error.message);
      } finally {
        setIsLoading(false);
      }
    },
    [options]
  );

  const clearImage = useCallback(() => {
    setImage(null);
    setError(null);
  }, []);

  return {
    image,
    pickImage,
    clearImage,
    isLoading,
    error,
  };
}


