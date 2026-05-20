import type { NavigatorScreenParams } from '@react-navigation/native';

export type RootStackParamList = {
  '(auth)': NavigatorScreenParams<AuthStackParamList>;
  '(tabs)': NavigatorScreenParams<TabParamList>;
  'video-detail': { videoId: string };
  'video-generation': undefined;
  'video-settings': { videoId?: string };
  'template-detail': { templateName: string };
  'batch-generation': undefined;
  'analytics': undefined;
  'search': undefined;
  'profile': undefined;
  'settings': undefined;
};

export type AuthStackParamList = {
  login: undefined;
  register: undefined;
  'forgot-password': undefined;
};

export type TabParamList = {
  index: undefined;
  'my-videos': undefined;
  'templates': undefined;
  'create': undefined;
  'analytics': undefined;
  'profile': undefined;
};

declare global {
  namespace ReactNavigation {
    interface RootParamList extends RootStackParamList {}
  }
}


