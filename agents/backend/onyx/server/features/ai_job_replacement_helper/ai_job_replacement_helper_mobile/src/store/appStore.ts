import { create } from 'zustand';
import type {
  GamificationProgress,
  DashboardData,
  Notification,
  Challenge,
} from '@/types';

interface AppState {
  // Gamification
  gamificationProgress: GamificationProgress | null;
  setGamificationProgress: (progress: GamificationProgress | null) => void;

  // Dashboard
  dashboardData: DashboardData | null;
  setDashboardData: (data: DashboardData | null) => void;

  // Notifications
  notifications: Notification[];
  unreadCount: number;
  setNotifications: (notifications: Notification[]) => void;
  setUnreadCount: (count: number) => void;
  addNotification: (notification: Notification) => void;
  markNotificationRead: (notificationId: string) => void;

  // Challenges
  activeChallenges: Challenge[];
  setActiveChallenges: (challenges: Challenge[]) => void;
  updateChallenge: (challenge: Challenge) => void;

  // UI State
  theme: 'light' | 'dark' | 'auto';
  setTheme: (theme: 'light' | 'dark' | 'auto') => void;
}

export const useAppStore = create<AppState>((set) => ({
  // Gamification
  gamificationProgress: null,
  setGamificationProgress: (progress) => set({ gamificationProgress: progress }),

  // Dashboard
  dashboardData: null,
  setDashboardData: (data) => set({ dashboardData: data }),

  // Notifications
  notifications: [],
  unreadCount: 0,
  setNotifications: (notifications) =>
    set({
      notifications,
      unreadCount: notifications.filter((n) => !n.read).length,
    }),
  setUnreadCount: (count) => set({ unreadCount: count }),
  addNotification: (notification) =>
    set((state) => ({
      notifications: [notification, ...state.notifications],
      unreadCount: state.unreadCount + (notification.read ? 0 : 1),
    })),
  markNotificationRead: (notificationId) =>
    set((state) => ({
      notifications: state.notifications.map((n) =>
        n.id === notificationId ? { ...n, read: true } : n
      ),
      unreadCount: Math.max(0, state.unreadCount - 1),
    })),

  // Challenges
  activeChallenges: [],
  setActiveChallenges: (challenges) => set({ activeChallenges: challenges }),
  updateChallenge: (challenge) =>
    set((state) => ({
      activeChallenges: state.activeChallenges.map((c) =>
        c.id === challenge.id ? challenge : c
      ),
    })),

  // UI State
  theme: 'auto',
  setTheme: (theme) => set({ theme }),
}));


