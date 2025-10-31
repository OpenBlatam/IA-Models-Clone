/**
 * PDF Variantes API - TypeScript SDK
 * Main entry point for TypeScript frontend integration
 * Version: 2.0.0
 */

export * from "./types";
export * from "./api-client";
export * from "./config";

// Re-export for convenience
import { PDFVariantesAPIClient, createClient } from "./api-client";
import { API_CONFIG } from "./config";

/**
 * Default client instance using environment configuration
 */
export const apiClient = createClient({
  baseURL: API_CONFIG.baseURL,
  apiKey: API_CONFIG.apiKey,
  timeout: API_CONFIG.timeout,
});

// Default export
export default PDFVariantesAPIClient;






