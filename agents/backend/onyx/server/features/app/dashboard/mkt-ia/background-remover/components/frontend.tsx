// services/frontend.ts
import { KeyMessageRequest, KeyMessageResponse, MessageType, MessageTone } from '../types/key-messages';

export class KeyMessageService {
  private baseUrl = '/api/key-messages';

  async generateMessage(request: KeyMessageRequest): Promise<KeyMessageResponse> {
    try {
      console.log('Sending request to:', `${this.baseUrl}/generate`);
      const response = await fetch(`${this.baseUrl}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: request.text,
          message_type: request.type,
          tone: request.tone,
          target_audience: request.target_audience,
          keywords: request.keywords || []
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || `Failed to generate message: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Received response:', data);
      return data;
    } catch (error) {
      console.error('Error in generateMessage:', error);
      throw error;
    }
  }

  async analyzeMessage(request: KeyMessageRequest): Promise<KeyMessageResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: request.text,
          message_type: request.type,
          tone: request.tone,
          target_audience: request.target_audience,
          keywords: request.keywords || []
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || `Failed to analyze message: ${response.statusText}`);
      }

      return response.json();
    } catch (error) {
      console.error('Error in analyzeMessage:', error);
      throw error;
    }
  }

  async getMessageTypes(): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseUrl}/types`);
      if (!response.ok) {
        throw new Error(`Failed to get message types: ${response.statusText}`);
      }
      return response.json();
    } catch (error) {
      console.error('Error in getMessageTypes:', error);
      throw error;
    }
  }

  async getMessageTones(): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseUrl}/tones`);
      if (!response.ok) {
        throw new Error(`Failed to get message tones: ${response.statusText}`);
      }
      return response.json();
    } catch (error) {
      console.error('Error in getMessageTones:', error);
      throw error;
    }
  }

  async clearCache(): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/cache`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        throw new Error(`Failed to clear cache: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Error in clearCache:', error);
      throw error;
    }
  }
}