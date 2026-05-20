import axios, { AxiosInstance, AxiosError } from 'axios';
import type {
  CalendarEvent,
  RoutineTask,
  Protocol,
  WardrobeItem,
  Outfit,
  DashboardData,
  DailySummary,
  WardrobeRecommendation,
  Alert,
  SearchRequest,
  Prediction,
  RoutineCompletion,
  ProtocolCompliance,
} from '@/types';
import { ApiError } from './api-error';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const BASE_PATH = '/artist-manager';

const createApiClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: `${API_URL}${BASE_PATH}`,
    headers: {
      'Content-Type': 'application/json',
    },
    timeout: 30000,
  });

  client.interceptors.response.use(
    (response) => response,
    (error: AxiosError) => {
      if (error.response) {
        const message =
          error.response.data?.detail ||
          error.response.data?.message ||
          error.message ||
          'Ha ocurrido un error';
        throw new ApiError(message, error.response.status, error.response.data);
      }
      if (error.request) {
        throw new ApiError('No se pudo conectar con el servidor', undefined, error.request);
      }
      throw new ApiError(error.message || 'Ha ocurrido un error inesperado');
    }
  );

  return client;
};

const api = createApiClient();

export const dashboardApi = {
  getDashboard: async (artistId: string): Promise<DashboardData> => {
    const response = await api.get(`/dashboard/${artistId}`);
    return response.data;
  },

  getDailySummary: async (artistId: string): Promise<DailySummary> => {
    const response = await api.get(`/dashboard/${artistId}/daily-summary`);
    return response.data;
  },
};

export const calendarApi = {
  createEvent: async (artistId: string, event: Omit<CalendarEvent, 'id'>): Promise<CalendarEvent> => {
    const response = await api.post(`/calendar/${artistId}/events`, event);
    return response.data;
  },

  getEvents: async (
    artistId: string,
    params?: { date?: string; days?: number }
  ): Promise<CalendarEvent[]> => {
    const response = await api.get(`/calendar/${artistId}/events`, { params });
    return response.data;
  },

  getEvent: async (artistId: string, eventId: string): Promise<CalendarEvent> => {
    const response = await api.get(`/calendar/${artistId}/events/${eventId}`);
    return response.data;
  },

  updateEvent: async (
    artistId: string,
    eventId: string,
    updates: Partial<CalendarEvent>
  ): Promise<CalendarEvent> => {
    const response = await api.put(`/calendar/${artistId}/events/${eventId}`, updates);
    return response.data;
  },

  deleteEvent: async (artistId: string, eventId: string): Promise<void> => {
    await api.delete(`/calendar/${artistId}/events/${eventId}`);
  },

  getWardrobeRecommendation: async (
    artistId: string,
    eventId: string
  ): Promise<WardrobeRecommendation> => {
    const response = await api.get(`/calendar/${artistId}/events/${eventId}/wardrobe-recommendation`);
    return response.data;
  },
};

export const routinesApi = {
  createRoutine: async (artistId: string, routine: Omit<RoutineTask, 'id'>): Promise<RoutineTask> => {
    const response = await api.post(`/routines/${artistId}/tasks`, routine);
    return response.data;
  },

  getRoutines: async (
    artistId: string,
    params?: { routine_type?: string; today_only?: boolean }
  ): Promise<RoutineTask[]> => {
    const response = await api.get(`/routines/${artistId}/tasks`, { params });
    return response.data;
  },

  getRoutine: async (artistId: string, taskId: string): Promise<RoutineTask> => {
    const response = await api.get(`/routines/${artistId}/tasks/${taskId}`);
    return response.data;
  },

  updateRoutine: async (
    artistId: string,
    taskId: string,
    updates: Partial<RoutineTask>
  ): Promise<RoutineTask> => {
    const response = await api.put(`/routines/${artistId}/tasks/${taskId}`, updates);
    return response.data;
  },

  deleteRoutine: async (artistId: string, taskId: string): Promise<void> => {
    await api.delete(`/routines/${artistId}/tasks/${taskId}`);
  },

  completeRoutine: async (
    artistId: string,
    taskId: string,
    completion: Partial<RoutineCompletion>
  ): Promise<RoutineCompletion> => {
    const response = await api.post(`/routines/${artistId}/tasks/${taskId}/complete`, completion);
    return response.data;
  },

  getCompletionRate: async (artistId: string, taskId: string, days = 30): Promise<number> => {
    const response = await api.get(`/routines/${artistId}/tasks/${taskId}/completion-rate`, {
      params: { days },
    });
    return response.data.completion_rate;
  },

  getPendingRoutines: async (artistId: string): Promise<RoutineTask[]> => {
    const response = await api.get(`/routines/${artistId}/pending`);
    return response.data;
  },
};

export const protocolsApi = {
  createProtocol: async (artistId: string, protocol: Omit<Protocol, 'id'>): Promise<Protocol> => {
    const response = await api.post(`/protocols/${artistId}`, protocol);
    return response.data;
  },

  getProtocols: async (
    artistId: string,
    params?: { category?: string; priority?: string; event_id?: string }
  ): Promise<Protocol[]> => {
    const response = await api.get(`/protocols/${artistId}`, { params });
    return response.data;
  },

  getProtocol: async (artistId: string, protocolId: string): Promise<Protocol> => {
    const response = await api.get(`/protocols/${artistId}/${protocolId}`);
    return response.data;
  },

  updateProtocol: async (
    artistId: string,
    protocolId: string,
    updates: Partial<Protocol>
  ): Promise<Protocol> => {
    const response = await api.put(`/protocols/${artistId}/${protocolId}`, updates);
    return response.data;
  },

  deleteProtocol: async (artistId: string, protocolId: string): Promise<void> => {
    await api.delete(`/protocols/${artistId}/${protocolId}`);
  },

  recordCompliance: async (
    artistId: string,
    protocolId: string,
    compliance: Omit<ProtocolCompliance, 'id' | 'checked_at'>
  ): Promise<ProtocolCompliance> => {
    const response = await api.post(`/protocols/${artistId}/${protocolId}/compliance`, compliance);
    return response.data;
  },

  checkEventCompliance: async (artistId: string, eventId: string): Promise<ProtocolCompliance> => {
    const response = await api.post(`/protocols/${artistId}/events/${eventId}/check-compliance`);
    return response.data;
  },

  getComplianceRate: async (artistId: string, protocolId: string): Promise<number> => {
    const response = await api.get(`/protocols/${artistId}/${protocolId}/compliance-rate`);
    return response.data.compliance_rate;
  },
};

export const wardrobeApi = {
  createItem: async (artistId: string, item: Omit<WardrobeItem, 'id'>): Promise<WardrobeItem> => {
    const response = await api.post(`/wardrobe/${artistId}/items`, item);
    return response.data;
  },

  getItems: async (
    artistId: string,
    params?: { category?: string; dress_code?: string }
  ): Promise<WardrobeItem[]> => {
    const response = await api.get(`/wardrobe/${artistId}/items`, { params });
    return response.data;
  },

  getItem: async (artistId: string, itemId: string): Promise<WardrobeItem> => {
    const response = await api.get(`/wardrobe/${artistId}/items/${itemId}`);
    return response.data;
  },

  updateItem: async (
    artistId: string,
    itemId: string,
    updates: Partial<WardrobeItem>
  ): Promise<WardrobeItem> => {
    const response = await api.put(`/wardrobe/${artistId}/items/${itemId}`, updates);
    return response.data;
  },

  deleteItem: async (artistId: string, itemId: string): Promise<void> => {
    await api.delete(`/wardrobe/${artistId}/items/${itemId}`);
  },

  createOutfit: async (artistId: string, outfit: Omit<Outfit, 'id'>): Promise<Outfit> => {
    const response = await api.post(`/wardrobe/${artistId}/outfits`, outfit);
    return response.data;
  },

  getOutfits: async (artistId: string, params?: { dress_code?: string }): Promise<Outfit[]> => {
    const response = await api.get(`/wardrobe/${artistId}/outfits`, { params });
    return response.data;
  },

  markItemWorn: async (artistId: string, itemId: string): Promise<WardrobeItem> => {
    const response = await api.post(`/wardrobe/${artistId}/items/${itemId}/mark-worn`);
    return response.data;
  },

  markOutfitWorn: async (artistId: string, outfitId: string): Promise<Outfit> => {
    const response = await api.post(`/wardrobe/${artistId}/outfits/${outfitId}/mark-worn`);
    return response.data;
  },
};

export const advancedApi = {
  searchEvents: async (artistId: string, searchRequest: SearchRequest): Promise<CalendarEvent[]> => {
    const response = await api.post(`/advanced/${artistId}/search/events`, searchRequest);
    return response.data;
  },

  getAlerts: async (artistId: string): Promise<Alert[]> => {
    const response = await api.get(`/advanced/${artistId}/alerts`);
    return response.data;
  },

  predictEventDuration: async (artistId: string, eventType: string): Promise<Prediction> => {
    const response = await api.get(`/advanced/${artistId}/predictions/event-duration`, {
      params: { event_type: eventType },
    });
    return response.data;
  },

  predictRoutineCompletion: async (
    artistId: string,
    routineId: string,
    days = 30
  ): Promise<Prediction> => {
    const response = await api.get(`/advanced/${artistId}/predictions/routine-completion`, {
      params: { routine_id: routineId, days },
    });
    return response.data;
  },

  syncCalendar: async (
    artistId: string,
    provider: string,
    credentials: Record<string, unknown>
  ): Promise<Record<string, unknown>> => {
    const response = await api.post(`/advanced/${artistId}/sync/calendar`, {
      provider,
      credentials,
    });
    return response.data;
  },
};

