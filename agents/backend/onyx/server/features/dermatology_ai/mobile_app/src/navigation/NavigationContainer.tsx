import React from 'react';
import { NavigationContainer as RNNavigationContainer } from '@react-navigation/native';
import { useDeepLinking } from '../hooks/useDeepLinking';
import { StackNavigator } from './StackNavigator';

/**
 * Component to handle deep linking inside NavigationContainer
 */
const DeepLinkingHandler: React.FC = React.memo(() => {
  useDeepLinking();
  return null;
});

DeepLinkingHandler.displayName = 'DeepLinkingHandler';

/**
 * Main Navigation Container with deep linking support
 */
export const AppNavigationContainer: React.FC = React.memo(() => {
  return (
    <RNNavigationContainer>
      <DeepLinkingHandler />
      <StackNavigator />
    </RNNavigationContainer>
  );
});

AppNavigationContainer.displayName = 'AppNavigationContainer';

