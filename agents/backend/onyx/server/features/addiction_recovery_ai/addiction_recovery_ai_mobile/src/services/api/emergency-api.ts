import { BaseApiClient } from './base-client';
import type { EmergencyContact } from '@/types';

export class EmergencyApi extends BaseApiClient {
  async createEmergencyContact(data: {
    user_id: string;
    name: string;
    phone: string;
    relationship: string;
    is_primary?: boolean;
  }): Promise<EmergencyContact> {
    const response = await this.client.post(
      this.getUrl('/emergency/contact'),
      data
    );
    return response.data;
  }

  async getEmergencyContacts(userId: string): Promise<EmergencyContact[]> {
    const response = await this.client.get(
      this.getUrl(`/emergency/contacts/${userId}`)
    );
    return response.data;
  }
}

export const emergencyApi = new EmergencyApi();

