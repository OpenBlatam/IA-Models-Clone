import { KeyMessageRequest, KeyMessageResponse } from './types/key-messages';
import { config } from './config';

export class KeyMessageService {
  private apiUrl: string;

  constructor() {
    this.apiUrl = config.api.baseUrl;
  }

  async generateMessage(request: KeyMessageRequest): Promise<KeyMessageResponse> {
    try {
      // Transform the request to match backend format
      const backendRequest = {
        message: request.text,
        message_type: request.type,
        tone: request.tone,
        target_audience: request.targetAudience,
        context: request.context,
        keywords: request.keywords || [],
        max_length: request.maxLength
      };

      console.log('Sending request to backend:', backendRequest);

      const response = await fetch(`${this.apiUrl}${config.api.endpoints.generate}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(backendRequest),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('Backend error response:', errorData);
        throw new Error(errorData.detail || errorData.error || config.errorMessages.generationError);
      }

      const backendResponse = await response.json();
      console.log('Received response from backend:', backendResponse);

      if (!backendResponse.success) {
        console.error('Backend response indicates failure:', backendResponse);
        throw new Error(backendResponse.error || config.errorMessages.generationError);
      }

      // Transform the backend response to match frontend format
      const transformedResponse = {
        success: true,
        data: backendResponse.data ? {
          id: backendResponse.data.id,
          originalMessage: backendResponse.data.original_message,
          response: backendResponse.data.response,
          messageType: backendResponse.data.message_type,
          tone: backendResponse.data.tone,
          createdAt: new Date().toISOString(),
          wordCount: backendResponse.data.word_count,
          characterCount: backendResponse.data.character_count,
          keywordsUsed: backendResponse.data.keywords_used || [],
          sentimentScore: backendResponse.data.sentiment_score,
          readabilityScore: backendResponse.data.readability_score
        } : undefined,
        error: undefined,
        processingTime: backendResponse.processing_time,
        suggestions: backendResponse.suggestions || []
      };
      
      console.log('Transformed response:', transformedResponse);
      console.log('Response text:', transformedResponse.data?.response);
      return transformedResponse;
    } catch (error) {
      console.error('Error in frontend service:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : config.errorMessages.generationError,
        processingTime: 0,
        suggestions: [],
      };
    }
  }
} 