/**
 * @deprecated This file is maintained for backward compatibility.
 * Please use the new modular API structure from '@/lib/api' instead.
 * 
 * This file will be removed in a future version.
 */

// Re-export from new API structure for backward compatibility
export {
  apiClient,
  postsApi,
  memesApi,
  calendarApi,
  platformsApi,
  analyticsApi,
  dashboardApi,
  templatesApi,
} from './api';

// Legacy exports with old method names for backward compatibility
import * as newPostsApi from './api/posts';
import * as newMemesApi from './api/memes';
import * as newCalendarApi from './api/calendar';
import * as newPlatformsApi from './api/platforms';
import * as newAnalyticsApi from './api/analytics';
import * as newDashboardApi from './api/dashboard';
import * as newTemplatesApi from './api/templates';

export const postsApi = {
  getAll: newPostsApi.getAllPosts,
  getById: newPostsApi.getPostById,
  create: newPostsApi.createPost,
  update: newPostsApi.updatePost,
  delete: newPostsApi.deletePost,
  publish: newPostsApi.publishPost,
  cancel: newPostsApi.cancelPost,
};

export const memesApi = {
  getAll: (category?: string, tags?: string, query?: string) =>
    newMemesApi.getAllMemes({ category, tags, query }),
  getById: newMemesApi.getMemeById,
  create: newMemesApi.createMeme,
  update: newMemesApi.updateMeme,
  delete: newMemesApi.deleteMeme,
  getRandom: newMemesApi.getRandomMeme,
};

export const calendarApi = {
  getEvents: newCalendarApi.getCalendarEvents,
  getDaily: newCalendarApi.getDailyEvents,
  getWeekly: newCalendarApi.getWeeklyEvents,
};

export const platformsApi = {
  getAll: newPlatformsApi.getAllPlatforms,
  connect: newPlatformsApi.connectPlatform,
  disconnect: newPlatformsApi.disconnectPlatform,
  getStatus: newPlatformsApi.getPlatformStatus,
};

export const analyticsApi = {
  getPlatformAnalytics: newAnalyticsApi.getPlatformAnalytics,
  getPostAnalytics: newAnalyticsApi.getPostAnalytics,
  getBestPerforming: newAnalyticsApi.getBestPerformingPosts,
  getTrends: newAnalyticsApi.getPlatformTrends,
};

export const dashboardApi = {
  getOverview: newDashboardApi.getDashboardOverview,
  getEngagement: newDashboardApi.getEngagementSummary,
  getUpcomingPosts: newDashboardApi.getUpcomingPosts,
  getRecentActivity: newDashboardApi.getRecentActivity,
};

export const templatesApi = {
  getAll: newTemplatesApi.getAllTemplates,
  getById: newTemplatesApi.getTemplateById,
  create: newTemplatesApi.createTemplate,
  update: newTemplatesApi.updateTemplate,
  delete: newTemplatesApi.deleteTemplate,
  render: newTemplatesApi.renderTemplate,
};

export { apiClient } from './api/client';


