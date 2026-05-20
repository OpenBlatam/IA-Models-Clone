import { useState, useCallback } from 'react';
import {
  checkPermission,
  requestPermission,
  requestPermissionWithAlert,
} from '@/utils/permissions';

type PermissionType = 'camera' | 'location' | 'notifications';

interface UsePermissionResult {
  granted: boolean;
  loading: boolean;
  check: () => Promise<void>;
  request: () => Promise<boolean>;
  requestWithAlert: (title: string, message: string) => Promise<boolean>;
}

export function usePermission(type: PermissionType): UsePermissionResult {
  const [granted, setGranted] = useState(false);
  const [loading, setLoading] = useState(false);

  const check = useCallback(async () => {
    setLoading(true);
    try {
      const result = await checkPermission(type);
      setGranted(result.granted);
    } catch (error) {
      console.error('Error checking permission:', error);
      setGranted(false);
    } finally {
      setLoading(false);
    }
  }, [type]);

  const request = useCallback(async () => {
    setLoading(true);
    try {
      const result = await requestPermission(type);
      setGranted(result.granted);
      return result.granted;
    } catch (error) {
      console.error('Error requesting permission:', error);
      setGranted(false);
      return false;
    } finally {
      setLoading(false);
    }
  }, [type]);

  const requestWithAlert = useCallback(
    async (title: string, message: string) => {
      setLoading(true);
      try {
        const result = await requestPermissionWithAlert(type, title, message);
        setGranted(result);
        return result;
      } catch (error) {
        console.error('Error requesting permission with alert:', error);
        setGranted(false);
        return false;
      } finally {
        setLoading(false);
      }
    },
    [type]
  );

  return {
    granted,
    loading,
    check,
    request,
    requestWithAlert,
  };
}

