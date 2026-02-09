import axios, { AxiosInstance, AxiosError } from 'axios';
import { API_BASE_URL, ENDPOINTS } from '@/constants/config';
import * as SecureStore from 'react-native-encrypted-storage';
import { STORAGE_KEYS } from '@/constants/config';
import type {
  User,
  Session,
  GamificationProgress,
  LeaderboardEntry,
  Roadmap,
  Step,
  Job,
  JobSwipeRequest,
  JobSwipeResponse,
  SkillRecommendation,
  JobRecommendation,
  Notification,
  MentoringSession,
  ChatMessage,
  CVAnalysis,
  InterviewSession,
  InterviewQuestion,
  InterviewResult,
  Challenge,
  DashboardData,
  ContentRequest,
  GeneratedContent,
  JobAlert,
  ApiResponse,
} from '@/types';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor to add auth token
    this.client.interceptors.request.use(
      async (config) => {
        const sessionId = await SecureStore.getItem(STORAGE_KEYS.SESSION_ID);
        if (sessionId && config.headers) {
          config.headers.Authorization = `Bearer ${sessionId}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        if (error.response?.status === 401) {
          // Clear session on unauthorized
          await SecureStore.removeItem(STORAGE_KEYS.SESSION_ID);
          await SecureStore.removeItem(STORAGE_KEYS.USER_ID);
          await SecureStore.removeItem(STORAGE_KEYS.USER_DATA);
        }
        return Promise.reject(error);
      }
    );
  }

  // Auth Methods
  async register(email: string, username: string, password: string): Promise<ApiResponse<User>> {
    try {
      const response = await this.client.post(ENDPOINTS.AUTH.REGISTER, null, {
        params: { email, username, password },
      });
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async login(email: string, password: string): Promise<ApiResponse<Session>> {
    try {
      const response = await this.client.post(ENDPOINTS.AUTH.LOGIN, null, {
        params: { email, password },
      });
      const session = response.data;
      if (session.session_id) {
        await SecureStore.setItem(STORAGE_KEYS.SESSION_ID, session.session_id);
        await SecureStore.setItem(STORAGE_KEYS.USER_ID, session.user_id);
        await SecureStore.setItem(STORAGE_KEYS.USER_DATA, JSON.stringify(session.user));
      }
      return { data: session };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async logout(): Promise<ApiResponse<{ success: boolean }>> {
    try {
      const sessionId = await SecureStore.getItem(STORAGE_KEYS.SESSION_ID);
      if (sessionId) {
        await this.client.post(ENDPOINTS.AUTH.LOGOUT, null, {
          params: { session_id: sessionId },
        });
      }
      await SecureStore.removeItem(STORAGE_KEYS.SESSION_ID);
      await SecureStore.removeItem(STORAGE_KEYS.USER_ID);
      await SecureStore.removeItem(STORAGE_KEYS.USER_DATA);
      return { data: { success: true } };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async verifySession(): Promise<ApiResponse<User>> {
    try {
      const sessionId = await SecureStore.getItem(STORAGE_KEYS.SESSION_ID);
      if (!sessionId) {
        return { error: 'No session found' };
      }
      const response = await this.client.get(`${ENDPOINTS.AUTH.VERIFY}/${sessionId}`);
      return { data: response.data.user };
    } catch (error) {
      return this.handleError(error);
    }
  }

  // Gamification Methods
  async getGamificationProgress(userId: string): Promise<ApiResponse<GamificationProgress>> {
    try {
      const response = await this.client.get(`${ENDPOINTS.GAMIFICATION.PROGRESS}/${userId}`);
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async addPoints(
    userId: string,
    action: string,
    amount?: number
  ): Promise<ApiResponse<GamificationProgress>> {
    try {
      const response = await this.client.post(`${ENDPOINTS.GAMIFICATION.POINTS}/${userId}`, null, {
        params: { action, amount },
      });
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async getLeaderboard(limit: number = 10): Promise<ApiResponse<LeaderboardEntry[]>> {
    try {
      const response = await this.client.get(ENDPOINTS.GAMIFICATION.LEADERBOARD, {
        params: { limit },
      });
      return { data: response.data.leaderboard || [] };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async getBadges(userId: string): Promise<ApiResponse<{ badges: any[]; total_badges: number }>> {
    try {
      const response = await this.client.get(`${ENDPOINTS.GAMIFICATION.BADGES}/${userId}`);
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  // Steps Methods
  async getRoadmap(userId: string): Promise<ApiResponse<Roadmap>> {
    try {
      const response = await this.client.get(`${ENDPOINTS.STEPS.ROADMAP}/${userId}`);
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async getStepsProgress(userId: string): Promise<ApiResponse<Roadmap>> {
    try {
      const response = await this.client.get(`${ENDPOINTS.STEPS.PROGRESS}/${userId}`);
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async startStep(userId: string, stepId: string): Promise<ApiResponse<Step>> {
    try {
      const response = await this.client.post(`${ENDPOINTS.STEPS.START}/${userId}`, {
        step_id: stepId,
      });
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async completeStep(
    userId: string,
    stepId: string,
    notes?: string
  ): Promise<ApiResponse<Step>> {
    try {
      const response = await this.client.post(`${ENDPOINTS.STEPS.COMPLETE}/${userId}`, {
        step_id: stepId,
        notes,
      });
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  // Jobs Methods
  async searchJobs(
    userId: string,
    params: {
      keywords?: string;
      location?: string;
      experience_level?: string;
      job_type?: string;
      limit?: number;
    }
  ): Promise<ApiResponse<{ jobs: Job[]; total: number }>> {
    try {
      const response = await this.client.get(`${ENDPOINTS.JOBS.SEARCH}/${userId}`, {
        params,
      });
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async swipeJob(userId: string, request: JobSwipeRequest): Promise<ApiResponse<JobSwipeResponse>> {
    try {
      const response = await this.client.post(`${ENDPOINTS.JOBS.SWIPE}/${userId}`, request);
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async applyToJob(
    userId: string,
    jobId: string,
    coverLetter?: string
  ): Promise<ApiResponse<{ success: boolean; message: string }>> {
    try {
      const response = await this.client.post(`${ENDPOINTS.JOBS.APPLY}/${userId}`, null, {
        params: { job_id: jobId, cover_letter: coverLetter },
      });
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async getSavedJobs(userId: string): Promise<ApiResponse<Job[]>> {
    try {
      const response = await this.client.get(`${ENDPOINTS.JOBS.SAVED}/${userId}`);
      return { data: response.data.jobs || [] };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async getLikedJobs(userId: string): Promise<ApiResponse<Job[]>> {
    try {
      const response = await this.client.get(`${ENDPOINTS.JOBS.LIKED}/${userId}`);
      return { data: response.data.jobs || [] };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async getMatches(userId: string): Promise<ApiResponse<Job[]>> {
    try {
      const response = await this.client.get(`${ENDPOINTS.JOBS.MATCHES}/${userId}`);
      return { data: response.data.matches || [] };
    } catch (error) {
      return this.handleError(error);
    }
  }

  // Recommendations Methods
  async getSkillRecommendations(
    userId: string,
    targetIndustry?: string
  ): Promise<ApiResponse<SkillRecommendation[]>> {
    try {
      const response = await this.client.get(`${ENDPOINTS.RECOMMENDATIONS.SKILLS}/${userId}`, {
        params: { target_industry: targetIndustry },
      });
      return { data: response.data.recommendations || [] };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async getJobRecommendations(
    userId: string,
    location?: string
  ): Promise<ApiResponse<JobRecommendation[]>> {
    try {
      const response = await this.client.get(`${ENDPOINTS.RECOMMENDATIONS.JOBS}/${userId}`, {
        params: { location },
      });
      return { data: response.data.recommendations || [] };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async getNextSteps(userId: string): Promise<ApiResponse<string[]>> {
    try {
      const response = await this.client.get(`${ENDPOINTS.RECOMMENDATIONS.NEXT_STEPS}/${userId}`);
      return { data: response.data.next_steps || [] };
    } catch (error) {
      return this.handleError(error);
    }
  }

  // Notifications Methods
  async getNotifications(
    userId: string,
    unreadOnly: boolean = false,
    limit: number = 20
  ): Promise<ApiResponse<Notification[]>> {
    try {
      const response = await this.client.get(`${ENDPOINTS.NOTIFICATIONS.LIST}/${userId}`, {
        params: { unread_only: unreadOnly, limit },
      });
      return { data: response.data.notifications || [] };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async getUnreadCount(userId: string): Promise<ApiResponse<number>> {
    try {
      const response = await this.client.get(
        `${ENDPOINTS.NOTIFICATIONS.UNREAD_COUNT}/${userId}`
      );
      return { data: response.data.count || 0 };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async markNotificationRead(
    userId: string,
    notificationId: string
  ): Promise<ApiResponse<{ success: boolean }>> {
    try {
      const response = await this.client.post(
        `${ENDPOINTS.NOTIFICATIONS.MARK_READ}/${userId}/${notificationId}`
      );
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async markAllNotificationsRead(userId: string): Promise<ApiResponse<{ success: boolean }>> {
    try {
      const response = await this.client.post(
        `${ENDPOINTS.NOTIFICATIONS.MARK_ALL_READ}/${userId}`
      );
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  // Mentoring Methods
  async startMentoringSession(
    userId: string,
    sessionType: string,
    mentorType: string
  ): Promise<ApiResponse<MentoringSession>> {
    try {
      const response = await this.client.post(`${ENDPOINTS.MENTORING.START}/${userId}`, null, {
        params: { session_type: sessionType, mentor_type: mentorType },
      });
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async askMentor(
    userId: string,
    sessionId: string,
    question: string
  ): Promise<ApiResponse<ChatMessage>> {
    try {
      const response = await this.client.post(
        `${ENDPOINTS.MENTORING.ASK}/${userId}/${sessionId}`,
        null,
        {
          params: { question },
        }
      );
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async getCareerAdvice(
    userId: string,
    currentSituation?: string,
    goals?: string
  ): Promise<ApiResponse<string>> {
    try {
      const response = await this.client.get(`${ENDPOINTS.MENTORING.CAREER_ADVICE}/${userId}`, {
        params: { current_situation: currentSituation, goals },
      });
      return { data: response.data.advice };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async getInterviewTips(
    userId: string,
    jobTitle?: string,
    company?: string
  ): Promise<ApiResponse<string>> {
    try {
      const response = await this.client.get(`${ENDPOINTS.MENTORING.INTERVIEW_TIPS}/${userId}`, {
        params: { job_title: jobTitle, company },
      });
      return { data: response.data.tips };
    } catch (error) {
      return this.handleError(error);
    }
  }

  // CV Analyzer Methods
  async analyzeCV(
    userId: string,
    cvContent: string,
    targetJob?: string
  ): Promise<ApiResponse<CVAnalysis>> {
    try {
      const response = await this.client.post(`${ENDPOINTS.CV.ANALYZE}/${userId}`, {
        cv_content: cvContent,
        target_job: targetJob,
      });
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  // Interview Methods
  async startInterview(
    userId: string,
    interviewType: string,
    jobTitle?: string,
    company?: string
  ): Promise<ApiResponse<InterviewSession>> {
    try {
      const response = await this.client.post(`${ENDPOINTS.INTERVIEW.START}/${userId}`, null, {
        params: { interview_type: interviewType, job_title: jobTitle, company },
      });
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async answerInterviewQuestion(
    userId: string,
    sessionId: string,
    questionId: string,
    answer: string
  ): Promise<ApiResponse<InterviewQuestion>> {
    try {
      const response = await this.client.post(
        `${ENDPOINTS.INTERVIEW.ANSWER}/${userId}/${sessionId}`,
        null,
        {
          params: { question_id: questionId, answer },
        }
      );
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async completeInterview(
    userId: string,
    sessionId: string
  ): Promise<ApiResponse<InterviewResult>> {
    try {
      const response = await this.client.post(
        `${ENDPOINTS.INTERVIEW.COMPLETE}/${userId}/${sessionId}`
      );
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  // Challenges Methods
  async getAvailableChallenges(
    userId: string,
    challengeType?: string
  ): Promise<ApiResponse<Challenge[]>> {
    try {
      const response = await this.client.get(`${ENDPOINTS.CHALLENGES.AVAILABLE}/${userId}`, {
        params: { challenge_type: challengeType },
      });
      return { data: response.data.challenges || [] };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async startChallenge(userId: string, challengeId: string): Promise<ApiResponse<Challenge>> {
    try {
      const response = await this.client.post(
        `${ENDPOINTS.CHALLENGES.START}/${userId}/${challengeId}`
      );
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async updateChallengeProgress(
    userId: string,
    challengeId: string,
    progress: number
  ): Promise<ApiResponse<Challenge>> {
    try {
      const response = await this.client.post(
        `${ENDPOINTS.CHALLENGES.PROGRESS}/${userId}/${challengeId}`,
        null,
        {
          params: { progress },
        }
      );
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async completeChallenge(userId: string, challengeId: string): Promise<ApiResponse<Challenge>> {
    try {
      const response = await this.client.post(
        `${ENDPOINTS.CHALLENGES.COMPLETE}/${userId}/${challengeId}`
      );
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  // Dashboard Methods
  async getDashboard(userId: string): Promise<ApiResponse<DashboardData>> {
    try {
      const response = await this.client.get(`${ENDPOINTS.DASHBOARD.MAIN}/${userId}`);
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async getDashboardMetrics(userId: string): Promise<ApiResponse<any>> {
    try {
      const response = await this.client.get(`${ENDPOINTS.DASHBOARD.METRICS}/${userId}`);
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async getActivityStats(userId: string, days: number = 30): Promise<ApiResponse<any>> {
    try {
      const response = await this.client.get(`${ENDPOINTS.DASHBOARD.ACTIVITY}/${userId}`, {
        params: { days },
      });
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  // Content Generator Methods
  async generateCoverLetter(request: ContentRequest): Promise<ApiResponse<GeneratedContent>> {
    try {
      const response = await this.client.post(ENDPOINTS.CONTENT.COVER_LETTER, request);
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async generateLinkedInPost(request: ContentRequest): Promise<ApiResponse<GeneratedContent>> {
    try {
      const response = await this.client.post(ENDPOINTS.CONTENT.LINKEDIN_POST, request);
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async generateFollowUpEmail(request: ContentRequest): Promise<ApiResponse<GeneratedContent>> {
    try {
      const response = await this.client.post(ENDPOINTS.CONTENT.FOLLOW_UP_EMAIL, request);
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async improveText(text: string, style?: string): Promise<ApiResponse<GeneratedContent>> {
    try {
      const response = await this.client.post(ENDPOINTS.CONTENT.IMPROVE_TEXT, {
        text,
        style,
      });
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  // Job Alerts Methods
  async createJobAlert(userId: string, alert: Partial<JobAlert>): Promise<ApiResponse<JobAlert>> {
    try {
      const response = await this.client.post(`${ENDPOINTS.JOB_ALERTS.CREATE}/${userId}`, alert);
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async getJobAlerts(userId: string): Promise<ApiResponse<JobAlert[]>> {
    try {
      const response = await this.client.get(`${ENDPOINTS.JOB_ALERTS.LIST}/${userId}`);
      return { data: response.data.alerts || [] };
    } catch (error) {
      return this.handleError(error);
    }
  }

  async checkJobAlerts(userId: string): Promise<ApiResponse<{ matches: Job[] }>> {
    try {
      const response = await this.client.post(`${ENDPOINTS.JOB_ALERTS.CHECK}/${userId}`);
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  // Health Check
  async healthCheck(): Promise<ApiResponse<any>> {
    try {
      const response = await this.client.get(ENDPOINTS.HEALTH.BASIC);
      return { data: response.data };
    } catch (error) {
      return this.handleError(error);
    }
  }

  // Error Handler
  private handleError(error: any): ApiResponse<any> {
    if (error.response) {
      return {
        error: error.response.data?.detail || error.response.data?.message || 'An error occurred',
      };
    } else if (error.request) {
      return { error: 'Network error. Please check your connection.' };
    } else {
      return { error: error.message || 'An unexpected error occurred' };
    }
  }
}

export const apiService = new ApiService();


