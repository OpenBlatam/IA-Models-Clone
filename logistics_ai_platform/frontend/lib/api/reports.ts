import { apiClient } from '../api-client';

export const reportsApi = {
  getDashboard: async (): Promise<any> => {
    const response = await apiClient.get('/reports/dashboard');
    return response.data;
  },

  getShipments: async (params?: any): Promise<any> => {
    const response = await apiClient.get('/reports/shipments', { params });
    return response.data;
  },
};




