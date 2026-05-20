/**
 * Stripe Provider Component
 * Isolated Stripe provider for better code splitting
 */

'use client';

import { Elements } from '@stripe/react-stripe-js';
import { getStripe } from '@/lib/stripe';
import type { ReactNode } from 'react';

interface StripeProviderProps {
  children: ReactNode;
}

/**
 * Stripe Elements provider
 */
export const StripeProvider = ({ children }: StripeProviderProps) => {
  const stripePromise = getStripe();

  if (!stripePromise) {
    return <>{children}</>;
  }

  return <Elements stripe={stripePromise}>{children}</Elements>;
};


