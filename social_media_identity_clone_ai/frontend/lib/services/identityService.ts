import { apiClient } from '@/lib/api/client';
import type { IdentityProfile, BuildIdentityRequest, BuildIdentityResponse } from '@/types';

export const identityService = {
  async getAll(): Promise<IdentityProfile[]> {
    try {
      return await apiClient.getIdentities();
    } catch {
      return await apiClient.searchIdentities({});
    }
  },

  async getById(id: string): Promise<IdentityProfile> {
    return apiClient.getIdentity(id);
  },

  async build(request: BuildIdentityRequest): Promise<BuildIdentityResponse> {
    return apiClient.buildIdentity(request);
  },

  async search(filters: unknown): Promise<IdentityProfile[]> {
    return apiClient.searchIdentities(filters);
  },
};



