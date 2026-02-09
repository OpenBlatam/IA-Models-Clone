import { useCallback } from 'react';
import * as Sharing from 'expo-sharing';
import { Alert, Platform } from 'react-native';

interface ShareOptions {
  message?: string;
  url?: string;
  title?: string;
}

/**
 * Hook for sharing content
 * Provides share functionality with error handling
 */
export function useShare() {
  const share = useCallback(async (options: ShareOptions) => {
    try {
      const isAvailable = await Sharing.isAvailableAsync();

      if (!isAvailable) {
        Alert.alert('Error', 'Sharing is not available on this device');
        return false;
      }

      if (options.url) {
        await Sharing.shareAsync(options.url, {
          mimeType: 'text/plain',
          dialogTitle: options.title || 'Share',
        });
      } else if (options.message) {
        // For text sharing, we need to use a different approach
        if (Platform.OS === 'ios') {
          await Sharing.shareAsync(options.message, {
            mimeType: 'text/plain',
            dialogTitle: options.title || 'Share',
          });
        } else {
          // Android - use Share API
          const { Share } = await import('react-native');
          await Share.share({
            message: options.message,
            title: options.title,
          });
        }
      }

      return true;
    } catch (error) {
      console.error('Share error:', error);
      Alert.alert('Error', 'Failed to share content');
      return false;
    }
  }, []);

  return { share };
}

