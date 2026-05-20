'use client';

import { useState } from 'react';
import { Layout } from '@/components/layout/Layout';
import { PricingCard } from '@/components/pricing/PricingCard';
import { useTranslations } from 'next-intl';
import { Plan } from '@/types/stripe';
import { stripeApi } from '@/lib/stripe-api';
import { useRouter } from '@/i18n/routing';
import { useLocale } from 'next-intl';
import { toast } from 'sonner';
import { Loading } from '@/components/ui/Loading';

const PLANS: Plan[] = [
  {
    id: 'free',
    name: 'Gratis',
    description: 'Perfecto para empezar',
    price: 0,
    currency: 'usd',
    interval: 'month',
    features: [
      'Hasta 10 posts por mes',
      '1 plataforma conectada',
      'Calendario básico',
      'Soporte por email',
    ],
    stripePriceId: '',
  },
  {
    id: 'pro',
    name: 'Pro',
    description: 'Para profesionales',
    price: 2999,
    currency: 'usd',
    interval: 'month',
    features: [
      'Posts ilimitados',
      'Todas las plataformas',
      'Calendario avanzado',
      'Analytics completos',
      'Soporte prioritario',
      'Plantillas premium',
    ],
    popular: true,
    stripePriceId: 'price_pro_monthly',
  },
  {
    id: 'enterprise',
    name: 'Enterprise',
    description: 'Para equipos',
    price: 9999,
    currency: 'usd',
    interval: 'month',
    features: [
      'Todo lo de Pro',
      'Múltiples usuarios',
      'API access',
      'Soporte 24/7',
      'Onboarding personalizado',
      'Custom integrations',
    ],
    stripePriceId: 'price_enterprise_monthly',
  },
];

export default function PricingPage() {
  const [loading, setLoading] = useState<string | null>(null);
  const router = useRouter();
  const locale = useLocale();
  const t = useTranslations('pricing');

  const handleSelectPlan = async (plan: Plan) => {
    if (plan.id === 'free') {
      toast.info('El plan gratuito ya está activo');
      return;
    }

    if (!plan.stripePriceId) {
      toast.error('Plan no disponible');
      return;
    }

    setLoading(plan.id);

    try {
      const successUrl = `${window.location.origin}/${locale}/pricing/success?session_id={CHECKOUT_SESSION_ID}`;
      const cancelUrl = `${window.location.origin}/${locale}/pricing`;

      const { sessionId } = await stripeApi.createCheckoutSession(
        plan.stripePriceId,
        successUrl,
        cancelUrl
      );

      const stripe = await import('@stripe/stripe-js').then((mod) => mod.loadStripe(
        process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY || ''
      ));

      if (!stripe) {
        throw new Error('Stripe no está configurado');
      }

      const { error } = await stripe.redirectToCheckout({ sessionId });

      if (error) {
        throw error;
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : t('error');
      toast.error(errorMessage);
      console.error('Error creating checkout session:', err);
    } finally {
      setLoading(null);
    }
  };

  return (
    <Layout>
      <div className="space-y-8">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-gray-100">{t('title')}</h1>
          <p className="mt-4 text-lg text-gray-600 dark:text-gray-400">{t('subtitle')}</p>
        </div>

        <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3">
          {PLANS.map((plan) => (
            <PricingCard
              key={plan.id}
              plan={plan}
              onSelect={handleSelectPlan}
              loading={loading === plan.id}
            />
          ))}
        </div>

        <div className="mt-12 text-center">
          <p className="text-sm text-gray-600 dark:text-gray-400">{t('note')}</p>
        </div>
      </div>
    </Layout>
  );
}



