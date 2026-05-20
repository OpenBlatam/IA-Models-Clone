import { apiClient } from '../api-client';

export const trackingApi = {
  track: async (identifier: string): Promise<any> => {
    const response = await apiClient.get(`/tracking/${identifier}`);
    return response.data;
  },

  getByShipmentId: async (shipmentId: string): Promise<any> => {
    const response = await apiClient.get(`/tracking/shipment/${shipmentId}`);
    return response.data;
  },

  getHistory: async (shipmentId: string): Promise<any[]> => {
    const response = await apiClient.get(`/tracking/shipment/${shipmentId}/history`);
    return response.data;
  },

  getSummary: async (): Promise<any> => {
    const response = await apiClient.get('/tracking/summary');
    return response.data;
  },
};




