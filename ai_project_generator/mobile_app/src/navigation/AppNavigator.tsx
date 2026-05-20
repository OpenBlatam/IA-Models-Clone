import React from 'react';
import { Text } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { HomeScreen } from '../screens/HomeScreen';
import { ProjectsScreen } from '../screens/ProjectsScreen';
import { GenerateScreen } from '../screens/GenerateScreen';
import { ProjectDetailScreen } from '../screens/ProjectDetailScreen';
import { SettingsScreen } from '../screens/SettingsScreen';

const Stack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

const HomeStack = () => (
  <Stack.Navigator>
    <Stack.Screen
      name="Home"
      component={HomeScreen}
      options={{ title: 'Inicio' }}
    />
  </Stack.Navigator>
);

const ProjectsStack = () => (
  <Stack.Navigator>
    <Stack.Screen
      name="ProjectsList"
      component={ProjectsScreen}
      options={{ title: 'Proyectos' }}
    />
    <Stack.Screen
      name="ProjectDetail"
      component={ProjectDetailScreen}
      options={{ title: 'Detalle del Proyecto' }}
    />
  </Stack.Navigator>
);

const GenerateStack = () => (
  <Stack.Navigator>
    <Stack.Screen
      name="Generate"
      component={GenerateScreen}
      options={{ title: 'Generar Proyecto' }}
    />
  </Stack.Navigator>
);

const SettingsStack = () => (
  <Stack.Navigator>
    <Stack.Screen
      name="Settings"
      component={SettingsScreen}
      options={{ title: 'Configuración' }}
    />
  </Stack.Navigator>
);

const MainTabs = () => (
  <Tab.Navigator
    screenOptions={{
      headerShown: false,
      tabBarActiveTintColor: '#3b82f6',
      tabBarInactiveTintColor: '#6b7280',
    }}
  >
    <Tab.Screen
      name="HomeTab"
      component={HomeStack}
      options={{
        tabBarLabel: 'Inicio',
        tabBarIcon: () => <Text>🏠</Text>,
      }}
    />
    <Tab.Screen
      name="ProjectsTab"
      component={ProjectsStack}
      options={{
        tabBarLabel: 'Proyectos',
        tabBarIcon: () => <Text>📋</Text>,
      }}
    />
    <Tab.Screen
      name="GenerateTab"
      component={GenerateStack}
      options={{
        tabBarLabel: 'Generar',
        tabBarIcon: () => <Text>➕</Text>,
      }}
    />
    <Tab.Screen
      name="SettingsTab"
      component={SettingsStack}
      options={{
        tabBarLabel: 'Config',
        tabBarIcon: () => <Text>⚙️</Text>,
      }}
    />
  </Tab.Navigator>
);

export const AppNavigator: React.FC = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        <Stack.Screen name="Main" component={MainTabs} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};


