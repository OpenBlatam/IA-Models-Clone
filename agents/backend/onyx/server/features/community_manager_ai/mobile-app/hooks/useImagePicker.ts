import { useState } from 'react';
import * as ImagePicker from 'expo-image-picker';
import { Alert } from 'react-native';

interface ImagePickerOptions {
  allowsEditing?: boolean;
  quality?: number;
  aspect?: [number, number];
  maxWidth?: number;
  maxHeight?: number;
}

export function useImagePicker() {
  const [loading, setLoading] = useState(false);

  const pickImage = async (options: ImagePickerOptions = {}) => {
    setLoading(true);
    try {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      
      if (status !== 'granted') {
        Alert.alert(
          'Permission Required',
          'Please grant camera roll permissions to select images'
        );
        return null;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: options.allowsEditing ?? true,
        quality: options.quality ?? 0.8,
        aspect: options.aspect,
      });

      if (result.canceled) {
        return null;
      }

      return result.assets[0];
    } catch (error) {
      console.error('Error picking image:', error);
      Alert.alert('Error', 'Failed to pick image');
      return null;
    } finally {
      setLoading(false);
    }
  };

  const takePhoto = async (options: ImagePickerOptions = {}) => {
    setLoading(true);
    try {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      
      if (status !== 'granted') {
        Alert.alert(
          'Permission Required',
          'Please grant camera permissions to take photos'
        );
        return null;
      }

      const result = await ImagePicker.launchCameraAsync({
        allowsEditing: options.allowsEditing ?? true,
        quality: options.quality ?? 0.8,
        aspect: options.aspect,
      });

      if (result.canceled) {
        return null;
      }

      return result.assets[0];
    } catch (error) {
      console.error('Error taking photo:', error);
      Alert.alert('Error', 'Failed to take photo');
      return null;
    } finally {
      setLoading(false);
    }
  };

  return { pickImage, takePhoto, loading };
}


