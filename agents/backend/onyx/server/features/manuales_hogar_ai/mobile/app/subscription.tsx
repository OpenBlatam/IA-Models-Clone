import { View, StyleSheet } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Stack } from 'expo-router';
import { useApp } from '@/lib/context/app-context';
import { SubscriptionPlans } from '@/components/subscription/subscription-plans';

export default function SubscriptionScreen() {
  const { state } = useApp();

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: state.colors.background }]} edges={['top']}>
      <Stack.Screen
        options={{
          title: 'Subscription',
          headerShown: true,
        }}
      />
      <SubscriptionPlans />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});




