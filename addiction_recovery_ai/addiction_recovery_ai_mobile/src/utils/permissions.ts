import { Platform, Alert, Linking } from 'react-native';
import { check, request, PERMISSIONS, RESULTS, Permission } from 'react-native-permissions';

interface PermissionResult {
  granted: boolean;
  blocked: boolean;
  unavailable: boolean;
}

function getPermissionType(type: 'camera' | 'location' | 'notifications'): Permission {
  if (Platform.OS === 'ios') {
    switch (type) {
      case 'camera':
        return PERMISSIONS.IOS.CAMERA;
      case 'location':
        return PERMISSIONS.IOS.LOCATION_WHEN_IN_USE;
      case 'notifications':
        return PERMISSIONS.IOS.NOTIFICATIONS;
      default:
        return PERMISSIONS.IOS.CAMERA;
    }
  } else {
    switch (type) {
      case 'camera':
        return PERMISSIONS.ANDROID.CAMERA;
      case 'location':
        return PERMISSIONS.ANDROID.ACCESS_FINE_LOCATION;
      case 'notifications':
        return PERMISSIONS.ANDROID.POST_NOTIFICATIONS;
      default:
        return PERMISSIONS.ANDROID.CAMERA;
    }
  }
}

export async function checkPermission(
  type: 'camera' | 'location' | 'notifications'
): Promise<PermissionResult> {
  const permission = getPermissionType(type);

  try {
    const result = await check(permission);

    return {
      granted: result === RESULTS.GRANTED,
      blocked: result === RESULTS.BLOCKED,
      unavailable: result === RESULTS.UNAVAILABLE,
    };
  } catch (error) {
    console.error('Error checking permission:', error);
    return {
      granted: false,
      blocked: false,
      unavailable: true,
    };
  }
}

export async function requestPermission(
  type: 'camera' | 'location' | 'notifications'
): Promise<PermissionResult> {
  const permission = getPermissionType(type);

  try {
    const result = await request(permission);

    return {
      granted: result === RESULTS.GRANTED,
      blocked: result === RESULTS.BLOCKED,
      unavailable: result === RESULTS.UNAVAILABLE,
    };
  } catch (error) {
    console.error('Error requesting permission:', error);
    return {
      granted: false,
      blocked: false,
      unavailable: true,
    };
  }
}

export async function requestPermissionWithAlert(
  type: 'camera' | 'location' | 'notifications',
  title: string,
  message: string
): Promise<boolean> {
  const { granted, blocked } = await requestPermission(type);

  if (granted) {
    return true;
  }

  if (blocked) {
    Alert.alert(
      title,
      message,
      [
        { text: 'Cancelar', style: 'cancel' },
        {
          text: 'Abrir Configuración',
          onPress: () => Linking.openSettings(),
        },
      ]
    );
  }

  return false;
}

