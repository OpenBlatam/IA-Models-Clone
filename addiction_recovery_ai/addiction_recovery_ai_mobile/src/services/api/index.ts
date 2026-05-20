// Base client
export { BaseApiClient } from './base-client';

// API modules
export { authApi, AuthApi } from './auth-api';
export { progressApi, ProgressApi } from './progress-api';
export { assessmentApi, AssessmentApi } from './assessment-api';
export { recoveryPlanApi, RecoveryPlanApi } from './recovery-plan-api';
export { relapsePreventionApi, RelapsePreventionApi } from './relapse-prevention-api';
export { supportApi, SupportApi } from './support-api';
export { analyticsApi, AnalyticsApi } from './analytics-api';
export { notificationsApi, NotificationsApi } from './notifications-api';
export { gamificationApi, GamificationApi } from './gamification-api';
export { dashboardApi, DashboardApi } from './dashboard-api';
export { chatbotApi, ChatbotApi } from './chatbot-api';
export { emergencyApi, EmergencyApi } from './emergency-api';
export { healthApi, HealthApi } from './health-api';

// Legacy API Service (for backward compatibility)
import { authApi } from './auth-api';
import { progressApi } from './progress-api';
import { assessmentApi } from './assessment-api';
import { recoveryPlanApi } from './recovery-plan-api';
import { relapsePreventionApi } from './relapse-prevention-api';
import { supportApi } from './support-api';
import { analyticsApi } from './analytics-api';
import { notificationsApi } from './notifications-api';
import { gamificationApi } from './gamification-api';
import { dashboardApi } from './dashboard-api';
import { chatbotApi } from './chatbot-api';
import { emergencyApi } from './emergency-api';
import { healthApi } from './health-api';
import type {
  RegisterRequest,
  RegisterResponse,
  LoginRequest,
  LoginResponse,
  AssessmentRequest,
  AssessmentResponse,
  ProfileResponse,
  UpdateProfileRequest,
  LogEntryRequest,
  LogEntryResponse,
  ProgressResponse,
  StatsResponse,
  TimelineResponse,
  CreateRecoveryPlanRequest,
  RecoveryPlanResponse,
  CheckRelapseRiskRequest,
  RelapseRiskResponse,
  TriggerResponse,
  CoachingSessionRequest,
  CoachingSessionResponse,
  MotivationResponse,
  AnalyticsResponse,
  Notification,
  PointsResponse,
  DashboardResponse,
  Achievement,
  Reminder,
  EmergencyContact,
} from '@/types';

class ApiService {
  // Token management
  async setToken(token: string): Promise<void> {
    return authApi.setToken(token);
  }

  async getToken(): Promise<string | null> {
    return authApi.getToken();
  }

  async clearToken(): Promise<void> {
    return authApi.clearToken();
  }

  // Auth endpoints
  async register(data: RegisterRequest): Promise<RegisterResponse> {
    return authApi.register(data);
  }

  async login(data: LoginRequest): Promise<LoginResponse> {
    return authApi.login(data);
  }

  async logout(): Promise<void> {
    return authApi.logout();
  }

  async getProfile(userId: string): Promise<ProfileResponse> {
    return authApi.getProfile(userId);
  }

  async updateProfile(data: UpdateProfileRequest): Promise<ProfileResponse> {
    return authApi.updateProfile(data);
  }

  // Assessment endpoints
  async assess(data: AssessmentRequest): Promise<AssessmentResponse> {
    return assessmentApi.assess(data);
  }

  // Progress endpoints
  async logEntry(data: LogEntryRequest): Promise<LogEntryResponse> {
    return progressApi.logEntry(data);
  }

  async getProgress(userId: string): Promise<ProgressResponse> {
    return progressApi.getProgress(userId);
  }

  async getStats(userId: string): Promise<StatsResponse> {
    return progressApi.getStats(userId);
  }

  async getTimeline(userId: string): Promise<TimelineResponse> {
    return progressApi.getTimeline(userId);
  }

  // Recovery Plan endpoints
  async createPlan(data: CreateRecoveryPlanRequest): Promise<RecoveryPlanResponse> {
    return recoveryPlanApi.createPlan(data);
  }

  async getPlan(userId: string): Promise<RecoveryPlanResponse> {
    return recoveryPlanApi.getPlan(userId);
  }

  async getStrategies(addictionType: string): Promise<string[]> {
    return recoveryPlanApi.getStrategies(addictionType);
  }

  // Relapse Prevention endpoints
  async checkRelapseRisk(data: CheckRelapseRiskRequest): Promise<RelapseRiskResponse> {
    return relapsePreventionApi.checkRelapseRisk(data);
  }

  async getTriggers(userId: string): Promise<TriggerResponse> {
    return relapsePreventionApi.getTriggers(userId);
  }

  async getCopingStrategies(data: {
    user_id: string;
    trigger?: string;
  }): Promise<string[]> {
    return relapsePreventionApi.getCopingStrategies(data);
  }

  // Support endpoints
  async coachingSession(data: CoachingSessionRequest): Promise<CoachingSessionResponse> {
    return supportApi.coachingSession(data);
  }

  async getMotivation(userId: string): Promise<MotivationResponse> {
    return supportApi.getMotivation(userId);
  }

  async celebrateMilestone(data: {
    user_id: string;
    milestone: string;
    days_sober: number;
  }): Promise<void> {
    return supportApi.celebrateMilestone(data);
  }

  async getAchievements(userId: string): Promise<Achievement[]> {
    return supportApi.getAchievements(userId);
  }

  // Analytics endpoints
  async getAnalytics(userId: string): Promise<AnalyticsResponse> {
    return analyticsApi.getAnalytics(userId);
  }

  async getAdvancedAnalytics(userId: string): Promise<AnalyticsResponse> {
    return analyticsApi.getAdvancedAnalytics(userId);
  }

  async getInsights(userId: string): Promise<string[]> {
    return analyticsApi.getInsights(userId);
  }

  // Notifications endpoints
  async getNotifications(userId: string): Promise<Notification[]> {
    return notificationsApi.getNotifications(userId);
  }

  async markNotificationRead(notificationId: string): Promise<void> {
    return notificationsApi.markNotificationRead(notificationId);
  }

  async getReminders(userId: string): Promise<Reminder[]> {
    return notificationsApi.getReminders(userId);
  }

  // Gamification endpoints
  async getPoints(userId: string): Promise<PointsResponse> {
    return gamificationApi.getPoints(userId);
  }

  async getGamificationAchievements(userId: string): Promise<Achievement[]> {
    return gamificationApi.getGamificationAchievements(userId);
  }

  // Dashboard endpoint
  async getDashboard(userId: string): Promise<DashboardResponse> {
    return dashboardApi.getDashboard(userId);
  }

  // Health check
  async healthCheck(): Promise<{ status: string; service: string; version: string }> {
    return healthApi.healthCheck();
  }

  // Chatbot endpoints
  async sendChatbotMessage(data: {
    user_id: string;
    message: string;
  }): Promise<{ response: string; session_id: string }> {
    return chatbotApi.sendChatbotMessage(data);
  }

  async startChatbotSession(data: {
    user_id: string;
  }): Promise<{ session_id: string }> {
    return chatbotApi.startChatbotSession(data);
  }

  // Emergency endpoints
  async createEmergencyContact(data: {
    user_id: string;
    name: string;
    phone: string;
    relationship: string;
    is_primary?: boolean;
  }): Promise<EmergencyContact> {
    return emergencyApi.createEmergencyContact(data);
  }

  async getEmergencyContacts(userId: string): Promise<EmergencyContact[]> {
    return emergencyApi.getEmergencyContacts(userId);
  }
}

export const apiService = new ApiService();

