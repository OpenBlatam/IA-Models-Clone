/**
 * Subscription Utilities
 * ======================
 * Utility functions for subscription management
 */

import type { Subscription, SubscriptionPlan } from '@/types/auth';

export function getSubscriptionFeatures(plan: SubscriptionPlan): string[] {
  const features: Record<SubscriptionPlan, string[]> = {
    free: [
      '5 manuales por mes',
      'Soporte básico',
      'Acceso a categorías básicas',
    ],
    basic: [
      '20 manuales por mes',
      'Soporte prioritario',
      'Todas las categorías',
      'Historial ilimitado',
    ],
    premium: [
      'Manuales ilimitados',
      'Soporte 24/7',
      'Todas las categorías',
      'Historial ilimitado',
      'Características avanzadas',
      'Sin límites de tokens',
    ],
  };

  return features[plan] || features.free;
}

export function getSubscriptionPrice(plan: SubscriptionPlan, interval: 'month' | 'year'): number {
  const prices: Record<SubscriptionPlan, { month: number; year: number }> = {
    free: { month: 0, year: 0 },
    basic: { month: 9.99, year: 99.99 },
    premium: { month: 19.99, year: 199.99 },
  };

  return prices[plan]?.[interval] || 0;
}

export function canGenerateManual(
  subscription: Subscription | null,
  manualsGeneratedThisMonth: number
): boolean {
  if (!subscription) {
    return manualsGeneratedThisMonth < 5; // Free tier limit
  }

  switch (subscription.plan) {
    case 'premium':
      return true; // Unlimited
    case 'basic':
      return manualsGeneratedThisMonth < 20;
    case 'free':
    default:
      return manualsGeneratedThisMonth < 5;
  }
}

export function getSubscriptionStatusColor(status: string): string {
  const colors: Record<string, string> = {
    active: '#34C759',
    inactive: '#8E8E93',
    expired: '#FF3B30',
    canceled: '#FF9500',
  };

  return colors[status] || colors.inactive;
}



