import { Share } from 'react-native';

interface ShareOptions {
  title?: string;
  message: string;
  url?: string;
}

export const useShare = () => {
  const share = async (options: ShareOptions) => {
    try {
      const result = await Share.share({
        title: options.title,
        message: options.message,
        url: options.url,
      });

      if (result.action === Share.sharedAction) {
        return { success: true, activityType: result.activityType };
      } else if (result.action === Share.dismissedAction) {
        return { success: false, dismissed: true };
      }
    } catch (error) {
      console.error('Error sharing:', error);
      return { success: false, error };
    }
  };

  return { share };
};

