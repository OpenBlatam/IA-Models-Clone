import React, { useCallback } from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import type { RouteProp } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import type { TabParamList } from '../types/navigation';
import { NAVIGATION_CONFIG, TAB_ICONS } from '../config/navigation';

// Feature modules
import { HomeScreen } from '../features/home';
import { CameraScreen } from '../features/camera';
import { HistoryScreen } from '../features/history';
import { ProfileScreen } from '../features/profile';

const Tab = createBottomTabNavigator<TabParamList>();

/**
 * Get tab icon based on route name and focus state
 */
const getTabIcon = (
  routeName: keyof typeof TAB_ICONS,
  focused: boolean
): keyof typeof Ionicons.glyphMap => {
  const icons = TAB_ICONS[routeName];
  if (!icons) {
    return TAB_ICONS.Profile.focused;
  }
  return focused ? icons.focused : icons.unfocused;
};

/**
 * Main Tab Navigator Component
 */
export const TabNavigator: React.FC = React.memo(() => {
  const tabScreenOptions = useCallback(
    ({ route }: { route: RouteProp<TabParamList, keyof TabParamList> }) => ({
      tabBarIcon: ({
        focused,
        color,
        size,
      }: {
        focused: boolean;
        color: string;
        size: number;
      }) => {
        const iconName = getTabIcon(route.name, focused);
        return <Ionicons name={iconName} size={size} color={color} />;
      },
      tabBarActiveTintColor: NAVIGATION_CONFIG.TAB.ACTIVE_COLOR,
      tabBarInactiveTintColor: NAVIGATION_CONFIG.TAB.INACTIVE_COLOR,
      headerShown: false,
    }),
    []
  );

  return (
    <Tab.Navigator screenOptions={tabScreenOptions}>
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen name="Camera" component={CameraScreen} />
      <Tab.Screen name="History" component={HistoryScreen} />
      <Tab.Screen name="Profile" component={ProfileScreen} />
    </Tab.Navigator>
  );
});

TabNavigator.displayName = 'TabNavigator';

