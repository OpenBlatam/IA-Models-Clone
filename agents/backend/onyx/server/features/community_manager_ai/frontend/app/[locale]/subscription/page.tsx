'use client';

import { useEffect, useState } from 'react';
import { Layout } from '@/components/layout/Layout';
import { Card, CardContent } from '@/components/ui/Card';
import { Loading } from '@/components/ui/Loading';
import { Alert } from '@/components/ui/Alert';
import { SubscriptionCard } from '@/components/subscription/SubscriptionCard';
import { Button } from '@/components/ui/Button';
import { useTranslations } from 'next-intl';
import { useRouter as useI18nRouter } from '@/i18n/routing';
import { Subscription } from '@/types/stripe';
import { stripeApi } from '@/lib/stripe-api';
import { CreditCard, ExternalLink } from 'lucide-react';

export default function SubscriptionPage() {
  const [subscription, setSubscription] = useState<Subscription | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const t = useTranslations('subscription');
  const router = useI18nRouter();

  useEffect(() => {
    fetchSubscription();
  }, []);

  const fetchSubscription = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await stripeApi.getSubscription();
      setSubscription(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : t('error');
      setError(errorMessage);
      console.error('Error fetching subscription:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleManageBilling = async () => {
    try {
      if (!subscription) return;

      const returnUrl = `${window.location.origin}${window.location.pathname}`;
      const { url } = await stripeApi.createPortalSession(
        subscription.id,
        returnUrl
      );

      if (url) {
        window.location.href = url;
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : t('error');
      console.error('Error creating portal session:', err);
    }
  };

  if (loading) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <Loading size="lg" text={t('loading')} />
        </div>
      </Layout>
    );
  }

  if (error && !subscription) {
    return (
      <Layout>
        <div className="space-y-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">{t('title')}</h1>
            <p className="mt-2 text-gray-600 dark:text-gray-400">{t('subtitle')}</p>
          </div>
          <Alert variant="error" title={t('error')}>
            {error}
          </Alert>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">{t('title')}</h1>
            <p className="mt-2 text-gray-600 dark:text-gray-400">{t('subtitle')}</p>
          </div>
          {subscription && (
            <Button
              variant="secondary"
              onClick={handleManageBilling}
              aria-label={t('manageBilling')}
            >
              <CreditCard className="mr-2 h-4 w-4" />
              {t('manageBilling')}
            </Button>
          )}
        </div>

        {subscription ? (
          <SubscriptionCard subscription={subscription} onUpdate={fetchSubscription} />
        ) : (
          <Card>
            <CardContent className="py-12 text-center">
              <p className="mb-4 text-gray-600 dark:text-gray-400">{t('noSubscription')}</p>
              <Button
                variant="primary"
                onClick={() => router.push('/pricing')}
                aria-label={t('viewPlans')}
              >
                {t('viewPlans')}
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    </Layout>
  );
}

