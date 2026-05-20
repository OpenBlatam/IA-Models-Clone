import { apiClient } from '../api-client';
import type { ShipmentRequest, ShipmentResponse, ShipmentStatus } from '@/types/api';

export const shipmentsApi = {
  create: async (data: ShipmentRequest): Promise<ShipmentResponse> => {
    const response = await apiClient.post<ShipmentResponse>('/forwarding/shipments', data);
    return response.data;
  },

  getById: async (shipmentId: string): Promise<ShipmentResponse> => {
    const response = await apiClient.get<ShipmentResponse>(`/forwarding/shipments/${shipmentId}`);
    return response.data;
  },

  getAll: async (params?: { status?: ShipmentStatus; limit?: number; offset?: number }): Promise<ShipmentResponse[]> => {
    const response = await apiClient.get<ShipmentResponse[]>('/forwarding/shipments', { params });
    return response.data;
  },

  updateStatus: async (
    shipmentId: string,
    data: { status: ShipmentStatus; location?: any; description?: string }
  ): Promise<ShipmentResponse> => {
    const response = await apiClient.patch<ShipmentResponse>(`/forwarding/shipments/${shipmentId}/status`, data);
    return response.data;
  },
};




