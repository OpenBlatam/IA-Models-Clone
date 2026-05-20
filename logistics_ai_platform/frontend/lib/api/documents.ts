import { apiClient } from '../api-client';
import type { DocumentResponse } from '@/types/api';

export const documentsApi = {
  upload: async (shipmentId: string, documentType: string, file: File, description?: string): Promise<DocumentResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('shipment_id', shipmentId);
    formData.append('document_type', documentType);
    if (description) {
      formData.append('description', description);
    }

    const response = await apiClient.post<DocumentResponse>('/documents', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  getById: async (documentId: string): Promise<DocumentResponse> => {
    const response = await apiClient.get<DocumentResponse>(`/documents/${documentId}`);
    return response.data;
  },

  getByShipment: async (shipmentId: string): Promise<DocumentResponse[]> => {
    const response = await apiClient.get<DocumentResponse[]>(`/documents/shipment/${shipmentId}`);
    return response.data;
  },

  delete: async (documentId: string): Promise<void> => {
    await apiClient.delete(`/documents/${documentId}`);
  },
};




