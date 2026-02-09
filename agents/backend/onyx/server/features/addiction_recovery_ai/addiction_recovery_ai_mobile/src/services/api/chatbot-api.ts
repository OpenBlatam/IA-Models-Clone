import { BaseApiClient } from './base-client';

export class ChatbotApi extends BaseApiClient {
  async sendChatbotMessage(data: {
    user_id: string;
    message: string;
  }): Promise<{ response: string; session_id: string }> {
    const response = await this.client.post(
      this.getUrl('/chatbot/message'),
      data
    );
    return response.data;
  }

  async startChatbotSession(data: {
    user_id: string;
  }): Promise<{ session_id: string }> {
    const response = await this.client.post(
      this.getUrl('/chatbot/start'),
      data
    );
    return response.data;
  }
}

export const chatbotApi = new ChatbotApi();

