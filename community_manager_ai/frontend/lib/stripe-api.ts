import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const stripeApi = {
  createCheckoutSession: async (priceId: string, successUrl: string, cancelUrl: string) => {
    const response = await axios.post(`${API_URL}/api/stripe/create-checkout-session`, {
      priceId,
      successUrl,
      cancelUrl,
    });
    return response.data;
  },

  createPortalSession: async (customerId: string, returnUrl: string) => {
    const response = await axios.post(`${API_URL}/api/stripe/create-portal-session`, {
      customerId,
      returnUrl,
    });
    return response.data;
  },

  getSubscription: async () => {
    const response = await axios.get(`${API_URL}/api/stripe/subscription`);
    return response.data;
  },

  cancelSubscription: async (subscriptionId: string) => {
    const response = await axios.post(`${API_URL}/api/stripe/cancel-subscription`, {
      subscriptionId,
    });
    return response.data;
  },
};



