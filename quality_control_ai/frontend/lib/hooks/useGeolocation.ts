import { useState, useEffect } from 'react';

interface GeolocationState {
  latitude: number | null;
  longitude: number | null;
  accuracy: number | null;
  error: GeolocationPositionError | null;
  loading: boolean;
}

interface UseGeolocationOptions {
  enableHighAccuracy?: boolean;
  timeout?: number;
  maximumAge?: number;
  watch?: boolean;
}

export const useGeolocation = (options: UseGeolocationOptions = {}) => {
  const {
    enableHighAccuracy = false,
    timeout = Infinity,
    maximumAge = 0,
    watch = false,
  } = options;

  const [state, setState] = useState<GeolocationState>({
    latitude: null,
    longitude: null,
    accuracy: null,
    error: null,
    loading: true,
  });

  useEffect(() => {
    if (!navigator.geolocation) {
      setState((prev) => ({
        ...prev,
        error: {
          code: 0,
          message: 'Geolocation is not supported',
          PERMISSION_DENIED: 1,
          POSITION_UNAVAILABLE: 2,
          TIMEOUT: 3,
        } as GeolocationPositionError,
        loading: false,
      }));
      return;
    }

    const onSuccess = (position: GeolocationPosition): void => {
      setState({
        latitude: position.coords.latitude,
        longitude: position.coords.longitude,
        accuracy: position.coords.accuracy,
        error: null,
        loading: false,
      });
    };

    const onError = (error: GeolocationPositionError): void => {
      setState((prev) => ({
        ...prev,
        error,
        loading: false,
      }));
    };

    const geoOptions: PositionOptions = {
      enableHighAccuracy,
      timeout,
      maximumAge,
    };

    let watchId: number | null = null;

    if (watch) {
      watchId = navigator.geolocation.watchPosition(onSuccess, onError, geoOptions);
    } else {
      navigator.geolocation.getCurrentPosition(onSuccess, onError, geoOptions);
    }

    return () => {
      if (watchId !== null) {
        navigator.geolocation.clearWatch(watchId);
      }
    };
  }, [enableHighAccuracy, timeout, maximumAge, watch]);

  return state;
};

