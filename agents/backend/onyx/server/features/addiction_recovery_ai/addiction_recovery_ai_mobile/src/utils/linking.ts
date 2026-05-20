import * as Linking from 'expo-linking';

export const linking = {
  prefixes: [Linking.createURL('/'), 'addictionrecoveryai://'],
  config: {
    screens: {
      Login: 'login',
      Register: 'register',
      Dashboard: 'dashboard',
      Progress: 'progress',
      Assessment: 'assessment',
    },
  },
};

export function useDeepLinking(): {
  openURL: (url: string) => Promise<void>;
  canOpenURL: (url: string) => Promise<boolean>;
  getInitialURL: () => Promise<string | null>;
} {
  const openURL = async (url: string): Promise<void> => {
    const canOpen = await Linking.canOpenURL(url);
    if (canOpen) {
      await Linking.openURL(url);
    } else {
      console.warn(`Cannot open URL: ${url}`);
    }
  };

  const canOpenURL = async (url: string): Promise<boolean> => {
    return await Linking.canOpenURL(url);
  };

  const getInitialURL = async (): Promise<string | null> => {
    return await Linking.getInitialURL();
  };

  return {
    openURL,
    canOpenURL,
    getInitialURL,
  };
}

