// Types for Robot Movement AI API

export interface Position {
  x: number;
  y: number;
  z: number;
  orientation?: number[];
}

export interface Waypoint {
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

export interface PathRequest {
  waypoints: Waypoint[];
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

export interface RobotStatus {
  robot_status: {
    connected: boolean;
    moving: boolean;
    position?: Position;
    joint_angles?: number[];
    velocity?: number[];
    error?: string;
  };
  feedback_stats: {
    frequency: number;
    total_samples: number;
    latest_timestamp?: number;
  };
  config: {
    robot_brand: string;
    ros_enabled: boolean;
    feedback_frequency: number;
  };
}

export interface Statistics {
  engine_statistics: Record<string, any>;
  chat_statistics: Record<string, any>;
  optimizer_statistics: Record<string, any>;
}

export interface Obstacle {
  min_x: number;
  min_y: number;
  min_z: number;
  max_x: number;
  max_y: number;
  max_z: number;
}

export interface TrajectoryAnalysis {
  total_distance: number;
  estimated_time: number;
  energy_consumption?: number;
  smoothness?: number;
  collision_free: boolean;
}

export interface HealthReport {
  status: string;
  timestamp: string;
  components: Record<string, any>;
  robot?: {
    initialized: boolean;
    status: any;
    feedback_stats: any;
  };
}

export interface Metrics {
  [key: string]: {
    value: number;
    timestamp: string;
    unit?: string;
  };
}

export interface Notification {
  id: string;
  type: string;
  message: string;
  timestamp: string;
  read: boolean;
}

export interface Task {
  id: string;
  name: string;
  status: string;
  created_at: string;
  updated_at: string;
}

