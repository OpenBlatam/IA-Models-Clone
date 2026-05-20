/**
 * Auth Types
 * ==========
 * TypeScript interfaces for authentication
 */

export interface User {
  id: string;
  email: string;
  name: string;
  picture?: string;
  created_at: string;
  subscription?: Subscription;
}

export interface Subscription {
  id: string;
  plan: SubscriptionPlan;
  status: SubscriptionStatus;
  current_period_start: string;
  current_period_end: string;
  cancel_at_period_end: boolean;
}

export type SubscriptionPlan = 'free' | 'basic' | 'premium';

export type SubscriptionStatus = 'active' | 'inactive' | 'expired' | 'canceled';

export interface AuthResponse {
  user: User;
  token: string;
}




