import { Tabs, useRouter, useSegments } from 'expo-router';
import { useColorScheme } from 'react-native';
import { Colors } from '@/constants/colors';
import { TabBarIcon } from '@/components/navigation/tab-bar-icon';
import { useAuthGuard } from '@/hooks/use-auth-guard';

export default function TabLayout() {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';
  
  // Protect routes - redirects to login if not authenticated
  useAuthGuard();

  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: isDark ? Colors.light.tint : Colors.dark.tint,
        tabBarInactiveTintColor: isDark ? Colors.light.tabIconDefault : Colors.dark.tabIconDefault,
        headerShown: false,
        tabBarStyle: {
          backgroundColor: isDark ? Colors.dark.tabBar : Colors.light.tabBar,
          borderTopColor: isDark ? Colors.dark.border : Colors.light.border,
        },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: 'Inicio',
          tabBarIcon: ({ color, focused }) => (
            <TabBarIcon name={focused ? 'home' : 'home-outline'} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="generate"
        options={{
          title: 'Generar',
          tabBarIcon: ({ color, focused }) => (
            <TabBarIcon name={focused ? 'add-circle' : 'add-circle-outline'} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="history"
        options={{
          title: 'Historial',
          tabBarIcon: ({ color, focused }) => (
            <TabBarIcon name={focused ? 'time' : 'time-outline'} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="profile"
        options={{
          title: 'Perfil',
          tabBarIcon: ({ color, focused }) => (
            <TabBarIcon name={focused ? 'person' : 'person-outline'} color={color} />
          ),
        }}
      />
    </Tabs>
  );
}

