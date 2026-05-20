import { apiClient } from '../api-client';
import type { ContainerRequest, ContainerResponse, ContainerStatus, GPSLocation } from '@/types/api';

export const containersApi = {
  create: async (data: ContainerRequest): Promise<ContainerResponse> => {
    const response = await apiClient.post<ContainerResponse>('/forwarding/containers', data);
    return response.data;
  },

  getAll: async (): Promise<ContainerResponse[]> => {
    try {
      const response = await apiClient.get<ContainerResponse[]>('/forwarding/containers');
      return response.data;
    } catch (error) {
      console.error('Error fetching containers:', error);
      return [];
    }
  },

  getById: async (containerId: string): Promise<ContainerResponse> => {
    const response = await apiClient.get<ContainerResponse>(`/forwarding/containers/${containerId}`);
    return response.data;
  },

  getByShipment: async (shipmentId: string): Promise<ContainerResponse[]> => {
    const response = await apiClient.get<ContainerResponse[]>(`/forwarding/containers/shipment/${shipmentId}`);
    return response.data;
  },

  updateStatus: async (containerId: string, status: ContainerStatus): Promise<ContainerResponse> => {
    const response = await apiClient.patch<ContainerResponse>(`/forwarding/containers/${containerId}/status`, { status });
    return response.data;
  },

  updateGPS: async (containerId: string, gpsLocation: GPSLocation): Promise<ContainerResponse> => {
    const response = await apiClient.patch<ContainerResponse>(`/forwarding/containers/${containerId}/gps`, gpsLocation);
    return response.data;
  },
};

