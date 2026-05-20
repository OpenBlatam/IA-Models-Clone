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
  text: string;
  type: MessageType;
  tone: MessageTone;
  targetAudience?: string;
  context?: string;
  keywords?: string[];
  maxLength?: number;
}

export interface GeneratedResponse {
  id: string;
  originalMessage: string;
  response: string;
  messageType: MessageType;
  tone: MessageTone;
  createdAt: string;
  wordCount: number;
  characterCount: number;
  keywordsUsed: string[];
  sentimentScore?: number;
  readabilityScore?: number;
}

export interface KeyMessageResponse {
  success: boolean;
  data?: GeneratedResponse;
  error?: string;
  processingTime: number;
  suggestions: string[];
} 