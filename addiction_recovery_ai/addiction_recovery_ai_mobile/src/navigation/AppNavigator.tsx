import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import { Ionicons } from '@expo/vector-icons';
import { useAuthStore } from '@/store/auth-store';
import { linking } from '@/utils/linking';
import { LazyComponentWrapper, createLazyComponent } from '@/hooks/use-lazy-component';

// Lazy load screens for code splitting
const LoginScreen = createLazyComponent(() => import('@/screens').then(m => ({ default: m.LoginScreen })));
const RegisterScreen = createLazyComponent(() => import('@/screens').then(m => ({ default: m.RegisterScreen })));
const DashboardScreen = createLazyComponent(() => import('@/screens').then(m => ({ default: m.DashboardScreen })));
const ProgressScreen = createLazyComponent(() => import('@/screens').then(m => ({ default: m.ProgressScreen })));
const AssessmentScreen = createLazyComponent(() => import('@/screens').then(m => ({ default: m.AssessmentScreen })));

const Tab = createBottomTabNavigator();
const Stack = createStackNavigator();

function MainTabs(): JSX.Element {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: keyof typeof Ionicons.glyphMap;

          if (route.name === 'Dashboard') {
            iconName = focused ? 'home' : 'home-outline';
          } else if (route.name === 'Progress') {
            iconName = focused ? 'stats-chart' : 'stats-chart-outline';
          } else if (route.name === 'Assessment') {
            iconName = focused ? 'clipboard' : 'clipboard-outline';
          } else {
            iconName = 'help-outline';
          }

          return <Ionicons name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: '#007AFF',
        tabBarInactiveTintColor: '#8E8E93',
        headerStyle: {
          backgroundColor: '#FFF',
        },
        headerTintColor: '#333',
        headerTitleStyle: {
          fontWeight: 'bold',
        },
      })}
    >
      <Tab.Screen
        name="Dashboard"
        component={DashboardScreen}
        options={{ title: 'Inicio' }}
      />
      <Tab.Screen
        name="Progress"
        component={ProgressScreen}
        options={{ title: 'Progreso' }}
      />
      <Tab.Screen
        name="Assessment"
        component={AssessmentScreen}
        options={{ title: 'Evaluación' }}
      />
    </Tab.Navigator>
  );
}

export function AppNavigator(): JSX.Element {
  const { isAuthenticated } = useAuthStore();

  return (
    <NavigationContainer linking={linking}>
      <LazyComponentWrapper>
        <Stack.Navigator screenOptions={{ headerShown: false }}>
          {!isAuthenticated ? (
            <>
              <Stack.Screen name="Login" component={LoginScreen} />
              <Stack.Screen name="Register" component={RegisterScreen} />
            </>
          ) : (
            <Stack.Screen name="Main" component={MainTabs} />
          )}
        </Stack.Navigator>
      </LazyComponentWrapper>
    </NavigationContainer>
  );
}
