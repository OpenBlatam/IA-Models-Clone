import { useState, useEffect } from 'react';
import { checkPermissions, requestCameraPermission, requestMediaLibraryPermission } from '../utils/permissions';

interface PermissionsState {
  camera: boolean;
  mediaLibrary: boolean;
  isLoading: boolean;
}

export const usePermissions = () => {
  const [permissions, setPermissions] = useState<PermissionsState>({
    camera: false,
    mediaLibrary: false,
    isLoading: true,
  });

  useEffect(() => {
    loadPermissions();
  }, []);

  const loadPermissions = async () => {
    try {
      const perms = await checkPermissions();
      setPermissions({
        camera: perms.camera,
        mediaLibrary: perms.mediaLibrary,
        isLoading: false,
      });
    } catch (error) {
      console.error('Error loading permissions:', error);
      setPermissions(prev => ({ ...prev, isLoading: false }));
    }
  };

  const requestCamera = async () => {
    const granted = await requestCameraPermission();
    if (granted) {
      setPermissions(prev => ({ ...prev, camera: true }));
    }
    return granted;
  };

  const requestMedia = async () => {
    const granted = await requestMediaLibraryPermission();
    if (granted) {
      setPermissions(prev => ({ ...prev, mediaLibrary: true }));
    }
    return granted;
  };

  const refresh = async () => {
    await loadPermissions();
  };

  return {
    ...permissions,
    requestCamera,
    requestMedia,
    refresh,
  };
};

