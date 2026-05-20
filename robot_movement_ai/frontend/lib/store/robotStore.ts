import { create } from 'zustand';
import { RobotStatus, Position, ChatResponse } from '../api/types';
import { apiClient } from '../api/client';
import { wsClient } from '../api/websocket';
import { toast } from '../utils/toast';

interface RobotState {
  // Status
  status: RobotStatus | null;
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;

  // Chat
  chatMessages: Array<{ role: 'user' | 'assistant'; content: string; timestamp: Date }>;
  isChatConnected: boolean;

  // Position
  currentPosition: Position | null;
  targetPosition: Position | null;

  // Actions
  fetchStatus: () => Promise<void>;
  moveTo: (position: Position) => Promise<void>;
  stop: () => Promise<void>;
  sendChatMessage: (message: string) => Promise<void>;
  connectWebSocket: () => Promise<void>;
  disconnectWebSocket: () => void;
  clearError: () => void;
}

export const useRobotStore = create<RobotState>((set, get) => ({
  status: null,
  isConnected: false,
  isLoading: false,
  error: null,
  chatMessages: [],
  isChatConnected: false,
  currentPosition: null,
  targetPosition: null,

  fetchStatus: async () => {
    set({ isLoading: true, error: null });
    try {
      const status = await apiClient.getStatus();
      set({
        status,
        isConnected: status.robot_status.connected,
        currentPosition: status.robot_status.position || null,
        isLoading: false,
      });
    } catch (error: any) {
      set({
        error: error.message || 'Failed to fetch status',
        isLoading: false,
      });
    }
  },

  moveTo: async (position: Position) => {
    set({ isLoading: true, error: null });
    try {
      await apiClient.moveTo(position);
      set({ targetPosition: position, isLoading: false });
      toast.success(`Robot moviéndose a (${position.x.toFixed(2)}, ${position.y.toFixed(2)}, ${position.z.toFixed(2)})`);
      // Refresh status after movement
      setTimeout(() => get().fetchStatus(), 1000);
    } catch (error: any) {
      const errorMsg = error.message || 'Failed to move robot';
      set({
        error: errorMsg,
        isLoading: false,
      });
      toast.error(errorMsg);
    }
  },

  stop: async () => {
    set({ isLoading: true, error: null });
    try {
      await apiClient.stop();
      set({ isLoading: false });
      toast.success('Robot detenido');
      get().fetchStatus();
    } catch (error: any) {
      const errorMsg = error.message || 'Failed to stop robot';
      set({
        error: errorMsg,
        isLoading: false,
      });
      toast.error(errorMsg);
    }
  },

  sendChatMessage: async (message: string) => {
    const userMessage = {
      role: 'user' as const,
      content: message,
      timestamp: new Date(),
    };

    set((state) => ({
      chatMessages: [...state.chatMessages, userMessage],
    }));

    try {
      if (get().isChatConnected && wsClient?.isConnected()) {
        wsClient.send(message);
      } else {
        const response = await apiClient.sendChatMessage(message);
        const assistantMessage = {
          role: 'assistant' as const,
          content: response.message || response.response || 'Respuesta recibida',
          timestamp: new Date(),
        };
        set((state) => ({
          chatMessages: [...state.chatMessages, assistantMessage],
        }));
      }
    } catch (error: any) {
      const errorMessage = {
        role: 'assistant' as const,
        content: `Error: ${error.message || 'Failed to send message'}`,
        timestamp: new Date(),
      };
      set((state) => ({
        chatMessages: [...state.chatMessages, errorMessage],
        error: error.message || 'Failed to send message',
      }));
    }
  },

  connectWebSocket: async () => {
    if (!wsClient) {
      set({ isChatConnected: false });
      return;
    }
    try {
      await wsClient.connect();
      wsClient.on('message', (response: ChatResponse) => {
        const assistantMessage = {
          role: 'assistant' as const,
          content: response.message || response.response || 'Mensaje recibido',
          timestamp: new Date(),
        };
        set((state) => ({
          chatMessages: [...state.chatMessages, assistantMessage],
        }));
      });
      set({ isChatConnected: true });
    } catch (error: any) {
      set({
        error: error.message || 'Failed to connect WebSocket',
        isChatConnected: false,
      });
    }
  },

  disconnectWebSocket: () => {
    wsClient.disconnect();
    set({ isChatConnected: false });
  },

  clearError: () => {
    set({ error: null });
  },
}));

