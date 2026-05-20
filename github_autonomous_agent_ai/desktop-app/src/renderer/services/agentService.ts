/**
 * Service for managing continuous agents
 */

import axios from 'axios';
import { APP_CONFIG } from '../../shared/config';
import type {
  ContinuousAgent,
  CreateAgentRequest,
  UpdateAgentRequest,
} from '../types/agent';

const API_BASE = `${APP_CONFIG.apiBaseUrl}/api/continuous-agent`;

class AgentService {
  private client = axios.create({
    baseURL: APP_CONFIG.apiBaseUrl,
    timeout: 30000,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  private getAuthHeaders() {
    const apiKey = typeof window !== 'undefined' 
      ? localStorage.getItem('api_key') 
      : null;
    
    return apiKey ? { Authorization: `Bearer ${apiKey}` } : {};
  }

  async fetchAgents(): Promise<ContinuousAgent[]> {
    const response = await this.client.get(API_BASE, {
      headers: this.getAuthHeaders(),
    });
    return response.data;
  }

  async fetchAgent(agentId: string): Promise<ContinuousAgent> {
    if (!agentId?.trim()) {
      throw new Error('Agent ID is required');
    }
    const response = await this.client.get(`${API_BASE}/${agentId}`, {
      headers: this.getAuthHeaders(),
    });
    return response.data;
  }

  async createAgent(request: CreateAgentRequest): Promise<ContinuousAgent> {
    const response = await this.client.post(API_BASE, request, {
      headers: this.getAuthHeaders(),
    });
    return response.data;
  }

  async updateAgent(
    agentId: string,
    request: UpdateAgentRequest
  ): Promise<ContinuousAgent> {
    if (!agentId?.trim()) {
      throw new Error('Agent ID is required');
    }
    const response = await this.client.patch(`${API_BASE}/${agentId}`, request, {
      headers: this.getAuthHeaders(),
    });
    return response.data;
  }

  async deleteAgent(agentId: string): Promise<void> {
    if (!agentId?.trim()) {
      throw new Error('Agent ID is required');
    }
    await this.client.delete(`${API_BASE}/${agentId}`, {
      headers: this.getAuthHeaders(),
    });
  }

  async toggleAgent(agentId: string): Promise<ContinuousAgent> {
    if (!agentId?.trim()) {
      throw new Error('Agent ID is required');
    }
    const response = await this.client.patch(
      `${API_BASE}/${agentId}/toggle`,
      {},
      {
        headers: this.getAuthHeaders(),
      }
    );
    return response.data;
  }
}

export const agentService = new AgentService();


