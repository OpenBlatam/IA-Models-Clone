import { useState, useEffect, useCallback } from 'react';
import * as Location from 'expo-location';
import { Alert } from 'react-native';

export interface LocationData {
  latitude: number;
  longitude: number;
  altitude: number | null;
  accuracy: number | null;
  heading: number | null;
  speed: number | null;
}

export function useLocation(options: Location.LocationOptions = {}) {
  const [location, setLocation] = useState<LocationData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);

  const requestPermission = useCallback(async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      const granted = status === 'granted';
      setHasPermission(granted);

      if (!granted) {
        Alert.alert('Permission required', 'Location permission is required');
      }

      return granted;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to request location permission');
      setError(error);
      return false;
    }
  }, []);

  const getCurrentLocation = useCallback(async () => {
    if (!hasPermission) {
      const granted = await requestPermission();
      if (!granted) return;
    }

    try {
      setIsLoading(true);
      setError(null);

      const currentLocation = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.Balanced,
        ...options,
      });

      setLocation({
        latitude: currentLocation.coords.latitude,
        longitude: currentLocation.coords.longitude,
        altitude: currentLocation.coords.altitude,
        accuracy: currentLocation.coords.accuracy,
        heading: currentLocation.coords.heading,
        speed: currentLocation.coords.speed,
      });
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to get location');
      setError(error);
      Alert.alert('Error', error.message);
    } finally {
      setIsLoading(false);
    }
  }, [hasPermission, requestPermission, options]);

  useEffect(() => {
    requestPermission();
  }, [requestPermission]);

  return {
    location,
    isLoading,
    error,
    hasPermission,
    getCurrentLocation,
    requestPermission,
  };
}


