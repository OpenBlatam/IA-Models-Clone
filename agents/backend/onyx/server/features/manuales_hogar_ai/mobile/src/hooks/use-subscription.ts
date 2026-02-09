/**
 * useSubscription Hook
 * ====================
 * Custom hook for subscription management
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useAuth } from '@/lib/context/auth-context';
import { subscriptionService } from '@/services/api/subscription-service';
import { Alert } from 'react-native';
import { useTranslation } from './use-translation';

export function useSubscription() {
  const { user } = useAuth();
  const { t } = useTranslation();
  const queryClient = useQueryClient();

  const { data: subscription, isLoading } = useQuery({
    queryKey: ['current-subscription'],
    queryFn: () => subscriptionService.getCurrentSubscription(),
    enabled: !!user,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  const { data: plans } = useQuery({
    queryKey: ['subscription-plans'],
    queryFn: () => subscriptionService.getPlans(),
    staleTime: 10 * 60 * 1000, // 10 minutes
  });

  const cancelMutation = useMutation({
    mutationFn: () => subscriptionService.cancelSubscription(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['current-subscription'] });
      queryClient.invalidateQueries({ queryKey: ['user'] });
      Alert.alert(t('common.success'), t('subscription.canceled'));
    },
    onError: () => {
      Alert.alert(t('common.error'), t('subscription.error'));
    },
  });

  const isActive = subscription?.status === 'active';
  const isPremium = subscription?.plan === 'premium';
  const isBasic = subscription?.plan === 'basic';
  const isFree = !subscription || subscription.plan === 'free';

  function cancelSubscription() {
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

  return {
    subscription,
    plans,
    isLoading,
    isActive,
    isPremium,
    isBasic,
    isFree,
    cancelSubscription,
    isCanceling: cancelMutation.isPending,
  };
}



