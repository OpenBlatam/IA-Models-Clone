'use client';

import { useState } from 'react';
import { loadStripe } from '@stripe/stripe-js';
import { Elements, PaymentElement, useStripe, useElements } from '@stripe/react-stripe-js';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import ErrorMessage from '@/components/ui/error-message';
import { createPaymentIntent } from '@/lib/stripe';
import { useTranslations } from 'next-intl';
import { getErrorMessage } from '@/lib/error-handler';

const getStripePromise = () => {
  const publishableKey = process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY;
  if (!publishableKey) {
    console.warn('Stripe publishable key is not configured');
    return null;
  }
  return loadStripe(publishableKey);
};

const stripePromise = getStripePromise();

interface StripeCheckoutProps {
  amount: number;
  currency?: string;
  metadata?: Record<string, string>;
  onSuccess?: () => void;
}

const CheckoutForm = ({ amount, currency = 'USD', metadata, onSuccess }: StripeCheckoutProps) => {
  const t = useTranslations();
  const stripe = useStripe();
  const elements = useElements();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!stripe || !elements) {
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const { clientSecret } = await createPaymentIntent(amount, currency, metadata);

      const { error: submitError } = await elements.submit();
      if (submitError) {
        setError(submitError.message || 'An error occurred');
        setIsLoading(false);
        return;
      }

      const { error: confirmError } = await stripe.confirmPayment({
        clientSecret,
        elements,
        confirmParams: {
          return_url: `${window.location.origin}/payment/success`,
        },
      });

      if (confirmError) {
        setError(confirmError.message || 'Payment failed');
      } else {
        onSuccess?.();
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4" noValidate>
      <PaymentElement />
      {error && <ErrorMessage message={error} id="payment-error" />}
      <Button
        type="submit"
        className="w-full"
        disabled={!stripe || isLoading}
        aria-label={t('invoices.payNow')}
        aria-describedby={error ? 'payment-error' : undefined}
      >
        {isLoading ? t('common.loading') : t('invoices.payNow')}
      </Button>
    </form>
  );
};

const StripeCheckout = (props: StripeCheckoutProps) => {
  if (!stripePromise) {
    return (
      <Card>
        <CardContent className="py-8 text-center">
          <p className="text-destructive">Stripe is not configured. Please contact support.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Payment</CardTitle>
      </CardHeader>
      <CardContent>
        <Elements stripe={stripePromise}>
          <CheckoutForm {...props} />
        </Elements>
      </CardContent>
    </Card>
  );
};

export default StripeCheckout;

