export enum MessageType {
  MARKETING = "marketing",
  EDUCATIONAL = "educational",
  PROMOTIONAL = "promotional",
  INFORMATIONAL = "informational",
  CALL_TO_ACTION = "call_to_action"
}

export enum MessageTone {
  PROFESSIONAL = "professional",
  CASUAL = "casual",
  FRIENDLY = "friendly",
  AUTHORITATIVE = "authoritative",
  CONVERSATIONAL = "conversational"
}

export interface KeyMessageRequest {
  message: string;
  message_type: MessageType;
  tone: MessageTone;
  target_audience?: string;
  context?: string;
  keywords?: string[];
  max_length?: number;
}

export interface GeneratedResponse {
  id: string;
  original_message: string;
  response: string;
  message_type: MessageType;
  tone: MessageTone;
  created_at: string;
  word_count: number;
  character_count: number;
  keywords_used: string[];
  sentiment_score?: number;
  readability_score?: number;
}

export interface KeyMessageResponse {
  success: boolean;
  data?: GeneratedResponse;
  error?: string;
  processing_time: number;
  suggestions: string[];
} 