import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { Provider } from 'react-redux';
import { store } from './src/store/store';
import { AppProviders } from './src/providers/AppProviders';
import { AppNavigationContainer } from './src/navigation/NavigationContainer';
import NetworkStatus from './src/components/NetworkStatus';

/**
 * Main App Component
 * Modular structure with separated concerns:
 * - Providers: Context providers and Redux store
 * - Navigation: Navigation configuration
 * - Components: Reusable UI components
 */
const App: React.FC = () => {
  return (
    <Provider store={store}>
      <AppProviders>
        <AppNavigationContainer />
        <StatusBar style="auto" />
        <NetworkStatus />
      </AppProviders>
    </Provider>
  );
};

export default App;

