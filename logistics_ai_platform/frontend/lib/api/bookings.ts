import { apiClient } from '../api-client';
import type { BookingRequest, BookingResponse } from '@/types/api';

export const bookingsApi = {
  create: async (data: BookingRequest): Promise<BookingResponse> => {
    const response = await apiClient.post<BookingResponse>('/forwarding/bookings', data);
    return response.data;
  },

  getAll: async (): Promise<BookingResponse[]> => {
    const response = await apiClient.get<BookingResponse[]>('/forwarding/bookings');
    return response.data;
  },

  getById: async (bookingId: string): Promise<BookingResponse> => {
    const response = await apiClient.get<BookingResponse>(`/forwarding/bookings/${bookingId}`);
    return response.data;
  },

  getByShipment: async (shipmentId: string): Promise<BookingResponse[]> => {
    const response = await apiClient.get<BookingResponse[]>(`/forwarding/bookings/shipment/${shipmentId}`);
    return response.data;
  },
};

