/**
 * Subscription Gate
 * =================
 * Component to gate features based on subscription
 */

import { ReactNode } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '@/lib/context/app-context';
import { useSubscription } from '@/hooks/use-subscription';
import { useTranslation } from '@/hooks/use-translation';

interface SubscriptionGateProps {
  children: ReactNode;
  requiredPlan?: 'basic' | 'premium';
  fallback?: ReactNode;
}

export function SubscriptionGate({
  children,
  requiredPlan = 'basic',
  fallback,
}: SubscriptionGateProps) {
  const { subscription, isPremium, isBasic, isFree } = useSubscription();
  const { state } = useApp();
  const router = useRouter();
  const { t } = useTranslation();
  const colors = state.colors;

  const hasAccess =
    isPremium ||
    (requiredPlan === 'basic' && (isBasic || isPremium)) ||
    (requiredPlan === 'premium' && isPremium);

  if (hasAccess) {
    return <>{children}</>;
  }

  if (fallback) {
    return <>{fallback}</>;
  }

  return (
    <View style={[styles.container, { backgroundColor: colors.card }]}>
      <Ionicons name="lock-closed" size={48} color={colors.textSecondary} />
      <Text style={[styles.title, { color: colors.text }]}>
        {t('subscription.upgradeRequired', 'Upgrade Required')}
      </Text>
      <Text style={[styles.message, { color: colors.textSecondary }]}>
        {t(
          'subscription.upgradeMessage',
          `This feature requires a ${requiredPlan} subscription. Upgrade now to unlock it!`
        )}
      </Text>
      <TouchableOpacity
        style={[styles.upgradeButton, { backgroundColor: colors.tint }]}
        onPress={() => router.push('/subscription')}
      >
        <Ionicons name="star" size={20} color="#FFFFFF" />
        <Text style={styles.upgradeButtonText}>{t('profile.upgrade')}</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 32,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 200,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    marginTop: 16,
    marginBottom: 8,
    textAlign: 'center',
  },
  message: {
    fontSize: 14,
    textAlign: 'center',
    marginBottom: 24,
    lineHeight: 20,
  },
  upgradeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
    gap: 8,
  },
  upgradeButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
});



