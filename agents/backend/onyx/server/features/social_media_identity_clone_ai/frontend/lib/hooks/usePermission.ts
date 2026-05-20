import { useState, useEffect } from 'react';

type PermissionName = 'camera' | 'microphone' | 'notifications' | 'geolocation' | 'persistent-storage' | 'push' | 'midi';

interface PermissionState {
  state: PermissionStatus['state'] | null;
  loading: boolean;
  error: Error | null;
}

export const usePermission = (permissionName: PermissionName): PermissionState => {
  const [state, setState] = useState<PermissionState>({
    state: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    if (typeof navigator === 'undefined' || !('permissions' in navigator)) {
      setState({
        state: null,
        loading: false,
        error: new Error('Permissions API not supported'),
      });
      return;
    }

    const queryPermission = async (): Promise<void> => {
      try {
        const result = await navigator.permissions.query({ name: permissionName as PermissionDescriptor['name'] });
        setState({
          state: result.state,
          loading: false,
          error: null,
        });

        result.onchange = () => {
          setState((prev) => ({
            ...prev,
            state: result.state,
          }));
        };
      } catch (error) {
        setState({
          state: null,
          loading: false,
          error: error instanceof Error ? error : new Error('Failed to query permission'),
        });
      }
    };

    queryPermission();
  }, [permissionName]);

  return state;
};



