/**
 * Stripe Configuration
 * ===================
 * Stripe setup for payments
 */

import { initStripe } from '@stripe/stripe-react-native';

const STRIPE_PUBLISHABLE_KEY =
  process.env.EXPO_PUBLIC_STRIPE_PUBLISHABLE_KEY || '';

export async function initializeStripe() {
  if (STRIPE_PUBLISHABLE_KEY) {
    await initStripe({
      publishableKey: STRIPE_PUBLISHABLE_KEY,
      merchantIdentifier: 'merchant.com.blatamacademy.manualeshogarai',
    });
  }
}

export { STRIPE_PUBLISHABLE_KEY };




