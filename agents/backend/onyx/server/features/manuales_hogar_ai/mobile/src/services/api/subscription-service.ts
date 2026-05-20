/**
 * Subscription Service
 * ====================
 * Service for managing subscriptions and payments
 */

import { apiClient } from './api-client';
import type { Subscription, SubscriptionPlan } from '@/types/auth';

export interface SubscriptionPlanDetails {
  id: string;
  name: string;
  plan: SubscriptionPlan;
  price: number;
  currency: string;
  interval: 'month' | 'year';
  features: string[];
  stripe_price_id: string;
}

export interface CreateSubscriptionRequest {
  plan_id: string;
  payment_method_id?: string;
}

export interface CreateSubscriptionResponse {
  subscription: Subscription;
  client_secret?: string; // For Stripe Payment Intent
}

export interface CancelSubscriptionResponse {
  success: boolean;
  message: string;
}

class SubscriptionService {
  /**
   * Get available subscription plans
   */
  async getPlans(): Promise<SubscriptionPlanDetails[]> {
    return apiClient.get<SubscriptionPlanDetails[]>('/api/v1/subscriptions/plans');
  }

  /**
   * Get current user subscription
   */
  async getCurrentSubscription(): Promise<Subscription | null> {
    try {
      return await apiClient.get<Subscription>('/api/v1/subscriptions/current');
    } catch (error) {
      return null;
    }
  }

  /**
   * Create subscription
   */
  async createSubscription(
    request: CreateSubscriptionRequest
  ): Promise<CreateSubscriptionResponse> {
    return apiClient.post<CreateSubscriptionResponse>('/api/v1/subscriptions', request);
  }

  /**
   * Cancel subscription
   */
  async cancelSubscription(): Promise<CancelSubscriptionResponse> {
    return apiClient.post<CancelSubscriptionResponse>('/api/v1/subscriptions/cancel');
  }

  /**
   * Update subscription plan
   */
  async updateSubscription(plan_id: string): Promise<Subscription> {
    return apiClient.put<Subscription>('/api/v1/subscriptions', { plan_id });
  }

  /**
   * Get subscription history
   */
  async getSubscriptionHistory(): Promise<Subscription[]> {
    return apiClient.get<Subscription[]>('/api/v1/subscriptions/history');
  }
}

export const subscriptionService = new SubscriptionService();




