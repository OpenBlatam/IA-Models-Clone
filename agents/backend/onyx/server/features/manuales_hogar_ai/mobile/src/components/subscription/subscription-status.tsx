/**
 * Subscription Status
 * ===================
 * Component to display subscription status
 */

import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '@/lib/context/app-context';
import { useSubscription } from '@/hooks/use-subscription';
import { useTranslation } from '@/hooks/use-translation';
import { getSubscriptionStatusColor } from '@/lib/utils/subscription-utils';

export function SubscriptionStatus() {
  const { state } = useApp();
  const { subscription, isActive, isPremium, isBasic, isFree } = useSubscription();
  const { t } = useTranslation();
  const router = useRouter();
  const colors = state.colors;

  if (!subscription && isFree) {
    return (
      <View style={[styles.container, { backgroundColor: colors.card }]}>
        <View style={styles.header}>
          <Ionicons name="star-outline" size={24} color={colors.textSecondary} />
          <Text style={[styles.title, { color: colors.text }]}>
            {t('subscription.free')}
          </Text>
        </View>
        <Text style={[styles.description, { color: colors.textSecondary }]}>
          {t('subscription.upgradeMessage', 'Upgrade to unlock more features')}
        </Text>
        <TouchableOpacity
          style={[styles.upgradeButton, { backgroundColor: colors.tint }]}
          onPress={() => router.push('/subscription')}
        >
          <Text style={styles.upgradeButtonText}>{t('profile.upgrade')}</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!subscription) return null;

  const statusColor = getSubscriptionStatusColor(subscription.status);

  return (
    <TouchableOpacity
      style={[styles.container, { backgroundColor: colors.card }]}
      onPress={() => router.push('/subscription')}
      activeOpacity={0.7}
    >
      <View style={styles.header}>
        <View style={styles.statusIndicator}>
          <View style={[styles.statusDot, { backgroundColor: statusColor }]} />
          <Text style={[styles.planName, { color: colors.text }]}>
            {t(`subscription.${subscription.plan}`)}
          </Text>
        </View>
        <Ionicons name="chevron-forward" size={20} color={colors.textSecondary} />
      </View>

      <View style={styles.details}>
        <Text style={[styles.statusText, { color: statusColor }]}>
          {t(`profile.${subscription.status}`)}
        </Text>
        {subscription.cancel_at_period_end && (
          <Text style={[styles.cancelText, { color: colors.warning }]}>
            {t('subscription.cancelsAt', 'Cancels at period end')}
          </Text>
        )}
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  statusIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  planName: {
    fontSize: 16,
    fontWeight: '600',
  },
  description: {
    fontSize: 14,
    marginBottom: 12,
  },
  details: {
    gap: 4,
  },
  statusText: {
    fontSize: 14,
    fontWeight: '500',
  },
  cancelText: {
    fontSize: 12,
  },
  upgradeButton: {
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 8,
  },
  upgradeButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
});



