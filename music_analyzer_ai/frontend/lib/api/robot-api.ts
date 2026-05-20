import axios from 'axios';

const ROBOT_API_URL = process.env.NEXT_PUBLIC_ROBOT_API_URL || 'http://localhost:8010';

const robotApi = axios.create({
  baseURL: ROBOT_API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface Position {
  x: number;
  y: number;
  z: number;
  orientation?: number[];
}

export interface MoveToRequest {
  x: number;
  y: number;
  z: number;
  orientation?: number[];
}

export interface ChatMessage {
  message: string;
  context?: Record<string, any>;
}

export interface ChatResponse {
  success: boolean;
  message: string;
  action?: string;
  data?: Record<string, any>;
}

export interface Waypoint {
  x: number;
  y: number;
  z: number;
  orientation?: number[];
}

export interface PathRequest {
  waypoints: Waypoint[];
  speed?: number;
  acceleration?: number;
}

export interface RobotStatus {
  connected: boolean;
  position?: Position;
  status?: string;
  battery?: number;
  temperature?: number;
  errors?: string[];
}

export interface TrajectoryResponse {
  success: boolean;
  trajectory_id?: string;
  waypoints?: Waypoint[];
  estimated_time?: number;
  distance?: number;
}

// API Functions
export const robotApiService = {
  // Health check
  healthCheck: async (): Promise<any> => {
    const response = await robotApi.get('/health');
    return response.data;
  },

  // Get robot status
  getStatus: async (): Promise<RobotStatus> => {
    const response = await robotApi.get('/status');
    return response.data;
  },

  // Connect robot
  connect: async (robotId?: string): Promise<any> => {
    const response = await robotApi.post('/connect', null, {
      params: robotId ? { robot_id: robotId } : {},
    });
    return response.data;
  },

  // Disconnect robot
  disconnect: async (): Promise<any> => {
    const response = await robotApi.post('/disconnect');
    return response.data;
  },

  // Move to position
  moveTo: async (position: MoveToRequest): Promise<any> => {
    const response = await robotApi.post('/move-to', position);
    return response.data;
  },

  // Chat with robot
  chat: async (message: string, context?: Record<string, any>): Promise<ChatResponse> => {
    const response = await robotApi.post<ChatResponse>('/chat', {
      message,
      context,
    });
    return response.data;
  },

  // Execute path
  executePath: async (path: PathRequest): Promise<TrajectoryResponse> => {
    const response = await robotApi.post<TrajectoryResponse>('/path', path);
    return response.data;
  },

  // Stop movement
  stop: async (): Promise<any> => {
    const response = await robotApi.post('/stop');
    return response.data;
  },

  // Emergency stop
  emergencyStop: async (): Promise<any> => {
    const response = await robotApi.post('/emergency-stop');
    return response.data;
  },

  // Get current position
  getPosition: async (): Promise<Position> => {
    const response = await robotApi.get<Position>('/position');
    return response.data;
  },

  // Get metrics
  getMetrics: async (): Promise<any> => {
    const response = await robotApi.get('/metrics');
    return response.data;
  },

  // Get tasks
  getTasks: async (): Promise<any> => {
    const response = await robotApi.get('/tasks');
    return response.data;
  },

  // Get analytics
  getAnalytics: async (): Promise<any> => {
    const response = await robotApi.get('/analytics');
    return response.data;
  },
};

