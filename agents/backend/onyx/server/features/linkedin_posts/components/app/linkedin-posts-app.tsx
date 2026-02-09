import React from 'react';
import { StatusBar } from 'react-native';
import LinkedInSafeAreaProvider from '../providers/safe-area-provider';
import LinkedInPostsScreen from '../screens/linkedin-posts-screen';

export function LinkedInPostsApp() {
  return (
    <LinkedInSafeAreaProvider>
      <StatusBar
        barStyle="dark-content"
        backgroundColor="transparent"
        translucent
      />
      <LinkedInPostsScreen />
    </LinkedInSafeAreaProvider>
  );
}

export default LinkedInPostsApp; 