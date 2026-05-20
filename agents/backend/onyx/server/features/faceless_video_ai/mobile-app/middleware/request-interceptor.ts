import { apiClient } from '@/utils/api-client';
import { useAuthStore } from '@/store/auth-store';
import { ErrorHandler } from './error-handler';

export function setupRequestInterceptors() {
  // Request interceptor is already set up in api-client.ts
  // This file can be used for additional interceptors or middleware

  // Example: Add request ID for tracking
  const originalGet = apiClient.get;
  const originalPost = apiClient.post;

  // You can add custom interceptors here if needed
  // For example, adding custom headers, logging, etc.
}

export function handleApiError(error: unknown): never {
  ErrorHandler.handle(error, { showAlert: false, logToSentry: true });
  throw error;
}


