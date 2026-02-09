import React, { useMemo, useCallback } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaProvider } from 'react-native-safe-area-context';

// Lazy load screens for performance
const LinkedInPostGenerator = React.lazy(() => import('../components/ui/linkedin-post-generator'));
const PostHistoryScreen = React.lazy(() => import('../components/screens/post-history-screen'));
const AnalyticsScreen = React.lazy(() => import('../components/screens/analytics-screen'));
const SettingsScreen = React.lazy(() => import('../components/screens/settings-screen'));

// Navigation types
type RootStackParamList = {
  Main: undefined;
  PostDetail: { postId: string };
  CreatePost: undefined;
};

type MainTabParamList = {
  Generator: undefined;
  History: undefined;
  Analytics: undefined;
  Settings: undefined;
};

const Stack = createStackNavigator<RootStackParamList>();
const Tab = createBottomTabNavigator<MainTabParamList>();

// Memoized tab navigator for performance
const MainTabNavigator = React.memo(() => {
  const tabBarIcon = useCallback(({ focused, color, size }: any, routeName: string) => {
    let iconName: keyof typeof Ionicons.glyphMap;

    switch (routeName) {
      case 'Generator':
        iconName = focused ? 'create' : 'create-outline';
        break;
      case 'History':
        iconName = focused ? 'time' : 'time-outline';
        break;
      case 'Analytics':
        iconName = focused ? 'analytics' : 'analytics-outline';
        break;
      case 'Settings':
        iconName = focused ? 'settings' : 'settings-outline';
        break;
      default:
        iconName = 'help-outline';
    }

    return <Ionicons name={iconName} size={size} color={color} />;
  }, []);

  const screenOptions = useMemo(() => ({
    tabBarActiveTintColor: '#007AFF',
    tabBarInactiveTintColor: '#8E8E93',
    tabBarStyle: {
      backgroundColor: '#fff',
      borderTopWidth: 1,
      borderTopColor: '#E5E5EA',
      paddingBottom: 5,
      paddingTop: 5,
      height: 60,
    },
    headerShown: false,
  }), []);

  return (
    <Tab.Navigator screenOptions={screenOptions}>
      <Tab.Screen
        name="Generator"
        component={LinkedInPostGenerator}
        options={{
          title: 'Generator',
          tabBarIcon: ({ focused, color, size }) => tabBarIcon({ focused, color, size }, 'Generator'),
        }}
      />
      <Tab.Screen
        name="History"
        component={PostHistoryScreen}
        options={{
          title: 'History',
          tabBarIcon: ({ focused, color, size }) => tabBarIcon({ focused, color, size }, 'History'),
        }}
      />
      <Tab.Screen
        name="Analytics"
        component={AnalyticsScreen}
        options={{
          title: 'Analytics',
          tabBarIcon: ({ focused, color, size }) => tabBarIcon({ focused, color, size }, 'Analytics'),
        }}
      />
      <Tab.Screen
        name="Settings"
        component={SettingsScreen}
        options={{
          title: 'Settings',
          tabBarIcon: ({ focused, color, size }) => tabBarIcon({ focused, color, size }, 'Settings'),
        }}
      />
    </Tab.Navigator>
  );
});

// Memoized stack navigator for performance
const RootStackNavigator = React.memo(() => {
  const screenOptions = useMemo(() => ({
    headerStyle: {
      backgroundColor: '#fff',
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 8,
      elevation: 4,
    },
    headerTitleStyle: {
      fontWeight: 'bold',
      fontSize: 18,
    },
    headerTintColor: '#007AFF',
  }), []);

  return (
    <Stack.Navigator screenOptions={screenOptions}>
      <Stack.Screen
        name="Main"
        component={MainTabNavigator}
        options={{ headerShown: false }}
      />
    </Stack.Navigator>
  );
});

// Main app component with performance optimizations
export function LinkedInPostsApp() {
  const navigationTheme = useMemo(() => ({
    dark: false,
    colors: {
      primary: '#007AFF',
      background: '#f5f5f5',
      card: '#fff',
      text: '#1a1a1a',
      border: '#E5E5EA',
      notification: '#FF3B30',
    },
  }), []);

  const onReady = useCallback(() => {
    // Navigation ready callback
    console.log('Navigation ready');
  }, []);

  const onStateChange = useCallback((state: any) => {
    // Navigation state change callback
    console.log('Navigation state changed:', state);
  }, []);

  return (
    <SafeAreaProvider>
      <NavigationContainer
        theme={navigationTheme}
        onReady={onReady}
        onStateChange={onStateChange}
      >
        <RootStackNavigator />
      </NavigationContainer>
    </SafeAreaProvider>
  );
}

// Export the app component
export default LinkedInPostsApp; 