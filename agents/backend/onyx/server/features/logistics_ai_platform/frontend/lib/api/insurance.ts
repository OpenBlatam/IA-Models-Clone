import { apiClient } from '../api-client';
import type { InsuranceRequest, InsuranceResponse } from '@/types/api';

export const insuranceApi = {
  create: async (data: InsuranceRequest): Promise<InsuranceResponse> => {
    const response = await apiClient.post<InsuranceResponse>('/insurance', data);
    return response.data;
  },

  getById: async (insuranceId: string): Promise<InsuranceResponse> => {
    const response = await apiClient.get<InsuranceResponse>(`/insurance/${insuranceId}`);
    return response.data;
  },

  getByShipment: async (shipmentId: string): Promise<InsuranceResponse> => {
    const response = await apiClient.get<InsuranceResponse>(`/insurance/shipment/${shipmentId}`);
    return response.data;
  },
};




