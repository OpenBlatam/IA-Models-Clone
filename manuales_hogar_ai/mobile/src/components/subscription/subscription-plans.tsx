/**
 * Subscription Plans
 * ==================
 * Component for displaying and selecting subscription plans
 */

import { View, Text, StyleSheet, TouchableOpacity, ScrollView } from 'react-native';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from '@/hooks/use-translation';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '@/lib/context/app-context';
import { useAuth } from '@/lib/context/auth-context';
import { subscriptionService } from '@/services/api/subscription-service';
import { useStripe } from '@stripe/stripe-react-native';
import { Alert } from 'react-native';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { ErrorMessage } from '@/components/ui/error-message';
import { getSubscriptionFeatures } from '@/lib/utils/subscription-utils';
import type { SubscriptionPlanDetails } from '@/services/api/subscription-service';

export function SubscriptionPlans() {
  const { t } = useTranslation();
  const { state } = useApp();
  const { user } = useAuth();
  const { initPaymentSheet, presentPaymentSheet } = useStripe();
  const queryClient = useQueryClient();
  const colors = state.colors;

  const { data: plans, isLoading, error } = useQuery({
    queryKey: ['subscription-plans'],
    queryFn: () => subscriptionService.getPlans(),
  });

  const { data: currentSubscription } = useQuery({
    queryKey: ['current-subscription'],
    queryFn: () => subscriptionService.getCurrentSubscription(),
    enabled: !!user,
  });

  const subscribeMutation = useMutation({
    mutationFn: async (planId: string) => {
      const response = await subscriptionService.createSubscription({ plan_id: planId });
      
      if (response.client_secret) {
        // Initialize payment sheet
        const { error: initError } = await initPaymentSheet({
          paymentIntentClientSecret: response.client_secret,
          merchantDisplayName: 'Manuales Hogar AI',
        });

        if (initError) {
          throw new Error(initError.message);
        }

        // Present payment sheet
        const { error: presentError } = await presentPaymentSheet();

        if (presentError) {
          throw new Error(presentError.message);
        }
      }

      return response;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['current-subscription'] });
      queryClient.invalidateQueries({ queryKey: ['user'] });
      Alert.alert(t('common.success'), t('subscription.success'));
    },
    onError: (error: Error) => {
      Alert.alert(t('common.error'), error.message || t('subscription.error'));
    },
  });

  const cancelMutation = useMutation({
    mutationFn: () => subscriptionService.cancelSubscription(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['current-subscription'] });
      Alert.alert(t('common.success'), t('subscription.canceled'));
    },
    onError: () => {
      Alert.alert(t('common.error'), t('subscription.error'));
    },
  });

  function handleSubscribe(planId: string) {
    if (!user) {
      Alert.alert(t('errors.error'), t('auth.notSignedIn'));
      return;
    }

    Alert.alert(
      t('subscription.selectPlan'),
      t('subscription.confirm', 'Are you sure you want to subscribe to this plan?'),
      [
        { text: t('common.cancel'), style: 'cancel' },
        {
          text: t('common.confirm'),
          onPress: () => subscribeMutation.mutate(planId),
        },
      ]
    );
  }

  function handleCancel() {
    Alert.alert(
      t('subscription.unsubscribe'),
      t('subscription.cancelConfirm'),
      [
        { text: t('common.cancel'), style: 'cancel' },
        {
          text: t('common.confirm'),
          style: 'destructive',
          onPress: () => cancelMutation.mutate(),
        },
      ]
    );
  }

  if (isLoading) {
    return <LoadingSpinner />;
  }

  if (error) {
    return <ErrorMessage message={t('errors.unknownError')} />;
  }

  if (!plans || plans.length === 0) {
    return null;
  }

  const currentPlanId = currentSubscription?.plan;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {plans.map((plan) => {
        const isCurrentPlan = currentPlanId === plan.plan;
        const isPopular = plan.plan === 'premium';

        return (
          <View
            key={plan.id}
            style={[
              styles.planCard,
              {
                backgroundColor: colors.card,
                borderColor: isCurrentPlan ? colors.tint : colors.border,
                borderWidth: isCurrentPlan ? 2 : 1,
              },
            ]}
          >
            {isPopular && (
              <View style={[styles.badge, { backgroundColor: colors.tint }]}>
                <Text style={styles.badgeText}>{t('subscription.mostPopular')}</Text>
              </View>
            )}

            <View style={styles.planHeader}>
              <Text style={[styles.planName, { color: colors.text }]}>
                {t(`subscription.${plan.plan}`)}
              </Text>
              <Text style={[styles.planPrice, { color: colors.tint }]}>
                {t('subscription.price', {
                  price: `${plan.currency} ${plan.price}`,
                  period: plan.interval === 'month' ? t('subscription.monthly') : t('subscription.yearly'),
                })}
              </Text>
            </View>

            <View style={styles.features}>
              {(plan.features.length > 0
                ? plan.features
                : getSubscriptionFeatures(plan.plan)
              ).map((feature, index) => (
                <View key={index} style={styles.feature}>
                  <Ionicons name="checkmark-circle" size={20} color={colors.success} />
                  <Text style={[styles.featureText, { color: colors.text }]}>{feature}</Text>
                </View>
              ))}
            </View>

            {isCurrentPlan ? (
              <View style={styles.currentPlan}>
                <Text style={[styles.currentPlanText, { color: colors.tint }]}>
                  {t('subscription.currentPlan')}
                </Text>
                <TouchableOpacity
                  style={[styles.cancelButton, { borderColor: colors.error }]}
                  onPress={handleCancel}
                >
                  <Text style={[styles.cancelButtonText, { color: colors.error }]}>
                    {t('subscription.unsubscribe')}
                  </Text>
                </TouchableOpacity>
              </View>
            ) : (
              <TouchableOpacity
                style={[styles.subscribeButton, { backgroundColor: colors.tint }]}
                onPress={() => handleSubscribe(plan.id)}
                disabled={subscribeMutation.isPending}
              >
                <Text style={styles.subscribeButtonText}>
                  {subscribeMutation.isPending
                    ? t('subscription.processing')
                    : t('subscription.subscribe')}
                </Text>
              </TouchableOpacity>
            )}
          </View>
        );
      })}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    padding: 20,
    gap: 16,
  },
  planCard: {
    padding: 20,
    borderRadius: 12,
    borderWidth: 1,
    position: 'relative',
  },
  badge: {
    position: 'absolute',
    top: -10,
    right: 20,
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  badgeText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '600',
  },
  planHeader: {
    marginBottom: 16,
  },
  planName: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  planPrice: {
    fontSize: 20,
    fontWeight: '600',
  },
  features: {
    marginBottom: 20,
    gap: 12,
  },
  feature: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  featureText: {
    fontSize: 14,
    flex: 1,
  },
  subscribeButton: {
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  subscribeButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  currentPlan: {
    alignItems: 'center',
    gap: 12,
  },
  currentPlanText: {
    fontSize: 14,
    fontWeight: '600',
  },
  cancelButton: {
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    width: '100%',
    alignItems: 'center',
  },
  cancelButtonText: {
    fontSize: 14,
    fontWeight: '600',
  },
});


