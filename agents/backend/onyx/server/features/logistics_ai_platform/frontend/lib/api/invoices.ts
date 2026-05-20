import { apiClient } from '../api-client';
import type { InvoiceRequest, InvoiceResponse } from '@/types/api';

export const invoicesApi = {
  create: async (data: InvoiceRequest): Promise<InvoiceResponse> => {
    const response = await apiClient.post<InvoiceResponse>('/invoices', data);
    return response.data;
  },

  getAll: async (params?: { limit?: number; offset?: number }): Promise<InvoiceResponse[]> => {
    const response = await apiClient.get<InvoiceResponse[]>('/invoices', { params });
    return response.data;
  },

  getById: async (invoiceId: string): Promise<InvoiceResponse> => {
    const response = await apiClient.get<InvoiceResponse>(`/invoices/${invoiceId}`);
    return response.data;
  },

  getByShipment: async (shipmentId: string): Promise<InvoiceResponse[]> => {
    const response = await apiClient.get<InvoiceResponse[]>(`/invoices/shipment/${shipmentId}`);
    return response.data;
  },
};




