'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Alert } from '@/components/ui/Alert';
import { Subscription } from '@/types/stripe';
import { useTranslations } from 'next-intl';
import { formatDate } from '@/lib/utils';
import { Calendar, CreditCard, X } from 'lucide-react';
import { useState } from 'react';
import { ConfirmDialog } from '@/components/ui/ConfirmDialog';
import { stripeApi } from '@/lib/stripe-api';
import { toast } from 'sonner';

interface SubscriptionCardProps {
  subscription: Subscription;
  onUpdate?: () => void;
}

export const SubscriptionCard = ({ subscription, onUpdate }: SubscriptionCardProps) => {
  const t = useTranslations('subscription');
  const [showCancelDialog, setShowCancelDialog] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleCancel = async () => {
    setLoading(true);
    try {
      await stripeApi.cancelSubscription(subscription.id);
      toast.success(t('canceled'));
      setShowCancelDialog(false);
      onUpdate?.();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : t('error');
      toast.error(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const getStatusBadge = () => {
    const statusMap = {
      active: { variant: 'success' as const, label: t('active') },
      canceled: { variant: 'error' as const, label: t('canceled') },
      past_due: { variant: 'warning' as const, label: t('pastDue') },
      unpaid: { variant: 'error' as const, label: t('unpaid') },
    };

    const status = statusMap[subscription.status] || statusMap.active;
    return <Badge variant={status.variant}>{status.label}</Badge>;
  };

  return (
    <>
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>{subscription.plan.name}</CardTitle>
            {getStatusBadge()}
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
              <Calendar className="h-4 w-4" />
              <span>
                {t('renewsOn')}: {formatDate(subscription.currentPeriodEnd)}
              </span>
            </div>
            {subscription.cancelAtPeriodEnd && (
              <Alert variant="warning" title={t('canceling')}>
                {t('willCancel')}
              </Alert>
            )}
          </div>

          {subscription.status === 'active' && !subscription.cancelAtPeriodEnd && (
            <Button
              variant="danger"
              size="sm"
              onClick={() => setShowCancelDialog(true)}
              aria-label={t('cancelSubscription')}
            >
              <X className="mr-2 h-4 w-4" />
              {t('cancelSubscription')}
            </Button>
          )}
        </CardContent>
      </Card>

      <ConfirmDialog
        isOpen={showCancelDialog}
        onClose={() => setShowCancelDialog(false)}
        onConfirm={handleCancel}
        title={t('cancelSubscription')}
        message={t('cancelConfirm')}
        confirmLabel={t('confirmCancel')}
        variant="danger"
        loading={loading}
      />
    </>
  );
};

