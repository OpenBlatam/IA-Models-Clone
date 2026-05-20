/**
 * API Module
 * Centralized export for all API functions
 * 
 * This module provides a clean interface to all API operations,
 * organized by feature domain.
 */

// Client
export { apiClient, createApiClient, apiGet, apiPost, apiPut, apiDelete } from './client';

// Feature APIs
export * as postsApi from './posts';
export * as memesApi from './memes';
export * as calendarApi from './calendar';
export * as platformsApi from './platforms';
export * as analyticsApi from './analytics';
export * as dashboardApi from './dashboard';
export * as templatesApi from './templates';


