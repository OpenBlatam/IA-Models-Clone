import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { QueryClient, QueryClientProvider } from 'react-query';

// Import screens
import HomeScreen from './screens/HomeScreen';
import ContentGenerationScreen from './screens/ContentGenerationScreen';
import ChatbotScreen from './screens/ChatbotScreen';
import VideoProcessingScreen from './screens/VideoProcessingScreen';
import IoTScreen from './screens/IoTScreen';
import QuantumComputingScreen from './screens/QuantumComputingScreen';
import AIServiceScreen from './screens/AIServiceScreen';
import AnalyticsScreen from './screens/AnalyticsScreen';
import WorkflowScreen from './screens/WorkflowScreen';
import MLScreen from './screens/MLScreen';
import AgentsScreen from './screens/AgentsScreen';
import BIScreen from './screens/BIScreen';
import CloudNativeScreen from './screens/CloudNativeScreen';
import SecurityScreen from './screens/SecurityScreen';
import BlockchainScreen from './screens/BlockchainScreen';
import BULScreen from './screens/BULScreen';
import MetaverseScreen from './screens/MetaverseScreen';
import ConsciousnessScreen from './screens/ConsciousnessScreen';
import QuantumAIScreen from './screens/QuantumAIScreen';
import NeuralInterfaceScreen from './screens/NeuralInterfaceScreen';
import SpaceExplorationScreen from './screens/SpaceExplorationScreen';
import DimensionHoppingScreen from './screens/DimensionHoppingScreen';
import ConsciousnessTranscendenceScreen from './screens/ConsciousnessTranscendenceScreen';
import InfiniteEvolutionScreen from './screens/InfiniteEvolutionScreen';
import TranscendentAbsoluteScreen from './screens/TranscendentAbsoluteScreen';
import UltimateCosmicScreen from './screens/UltimateCosmicScreen';
import EternalOmnipotentScreen from './screens/EternalOmnipotentScreen';
import InfiniteDivineScreen from './screens/InfiniteDivineScreen';

const Stack = createNativeStackNavigator();
const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <SafeAreaProvider>
        <NavigationContainer>
          <Stack.Navigator initialRouteName="Home">
            <Stack.Screen name="Home" component={HomeScreen} options={{ title: 'Gamma App' }} />
            <Stack.Screen name="ContentGeneration" component={ContentGenerationScreen} options={{ title: 'Generate Content' }} />
            <Stack.Screen name="Chatbot" component={ChatbotScreen} options={{ title: 'AI Chatbot' }} />
            <Stack.Screen name="VideoProcessing" component={VideoProcessingScreen} options={{ title: 'Video Processing' }} />
            <Stack.Screen name="IoT" component={IoTScreen} options={{ title: 'IoT & Edge' }} />
            <Stack.Screen name="QuantumComputing" component={QuantumComputingScreen} options={{ title: 'Quantum Computing' }} />
            <Stack.Screen name="AIService" component={AIServiceScreen} options={{ title: 'Advanced AI' }} />
            <Stack.Screen name="Analytics" component={AnalyticsScreen} options={{ title: 'Advanced Analytics' }} />
            <Stack.Screen name="Workflow" component={WorkflowScreen} options={{ title: 'Workflow Automation' }} />
            <Stack.Screen name="ML" component={MLScreen} options={{ title: 'Machine Learning' }} />
            <Stack.Screen name="Agents" component={AgentsScreen} options={{ title: 'AI Agents' }} />
            <Stack.Screen name="BI" component={BIScreen} options={{ title: 'Business Intelligence' }} />
            <Stack.Screen name="CloudNative" component={CloudNativeScreen} options={{ title: 'Cloud-Native' }} />
            <Stack.Screen name="Security" component={SecurityScreen} options={{ title: 'Advanced Security' }} />
            <Stack.Screen name="Blockchain" component={BlockchainScreen} options={{ title: 'Blockchain & NFTs' }} />
            <Stack.Screen name="BUL" component={BULScreen} options={{ title: 'BUL - Document Generation' }} />
            <Stack.Screen name="Metaverse" component={MetaverseScreen} options={{ title: 'Metaverse AR/VR' }} />
            <Stack.Screen name="Consciousness" component={ConsciousnessScreen} options={{ title: 'Consciousness AI' }} />
            <Stack.Screen name="QuantumAI" component={QuantumAIScreen} options={{ title: 'Quantum AI' }} />
            <Stack.Screen name="NeuralInterface" component={NeuralInterfaceScreen} options={{ title: 'Neural Interface' }} />
            <Stack.Screen name="SpaceExploration" component={SpaceExplorationScreen} options={{ title: 'Space Exploration' }} />
            <Stack.Screen name="DimensionHopping" component={DimensionHoppingScreen} options={{ title: 'Dimension Hopping' }} />
            <Stack.Screen name="ConsciousnessTranscendence" component={ConsciousnessTranscendenceScreen} options={{ title: 'Consciousness Transcendence' }} />
            <Stack.Screen name="InfiniteEvolution" component={InfiniteEvolutionScreen} options={{ title: 'Infinite Evolution' }} />
            <Stack.Screen name="TranscendentAbsolute" component={TranscendentAbsoluteScreen} options={{ title: 'Transcendent Absolute' }} />
            <Stack.Screen name="UltimateCosmic" component={UltimateCosmicScreen} options={{ title: 'Ultimate Cosmic' }} />
            <Stack.Screen name="EternalOmnipotent" component={EternalOmnipotentScreen} options={{ title: 'Eternal Omnipotent' }} />
            <Stack.Screen name="InfiniteDivine" component={InfiniteDivineScreen} options={{ title: 'Infinite Divine' }} />
          </Stack.Navigator>
        </NavigationContainer>
        <StatusBar style="auto" />
      </SafeAreaProvider>
    </QueryClientProvider>
  );
}