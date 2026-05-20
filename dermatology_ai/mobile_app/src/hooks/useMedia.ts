import { useState, useEffect } from 'react';
import * as ImagePicker from 'expo-image-picker';
import { MediaTypeOptions } from 'expo-image-picker';

interface UseMediaOptions {
  allowsEditing?: boolean;
  quality?: number;
  mediaTypes?: MediaTypeOptions;
}

export const useMedia = (options: UseMediaOptions = {}) => {
  const [media, setMedia] = useState<ImagePicker.ImagePickerResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const pickImage = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: options.mediaTypes || MediaTypeOptions.Images,
        allowsEditing: options.allowsEditing || false,
        quality: options.quality || 1,
      });

      if (!result.canceled) {
        setMedia(result);
      }
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error);
    } finally {
      setIsLoading(false);
    }
  };

  const takePhoto = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: options.mediaTypes || MediaTypeOptions.Images,
        allowsEditing: options.allowsEditing || false,
        quality: options.quality || 1,
      });

      if (!result.canceled) {
        setMedia(result);
      }
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error);
    } finally {
      setIsLoading(false);
    }
  };

  const clearMedia = () => {
    setMedia(null);
    setError(null);
  };

  return {
    media,
    pickImage,
    takePhoto,
    clearMedia,
    isLoading,
    error,
  };
};

