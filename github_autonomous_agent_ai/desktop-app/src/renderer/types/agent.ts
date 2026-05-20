/**
 * Type definitions for Continuous Agent
 */

export type TaskType = 'code_review' | 'bug_fix' | 'feature' | 'refactor' | 'documentation' | 'test';

export interface AgentConfig {
  taskType: TaskType;
  frequency: number; // in seconds
  parameters: Record<string, any>;
  goal?: string;
}

export interface AgentStats {
  totalExecutions: number;
  successfulExecutions: number;
  failedExecutions: number;
  lastExecution?: string;
  averageExecutionTime?: number;
}

export interface ContinuousAgent {
  id: string;
  name: string;
  description: string;
  isActive: boolean;
  config: AgentConfig;
  stats?: AgentStats;
  createdAt: string;
  updatedAt: string;
}

export interface CreateAgentRequest {
  name: string;
  description: string;
  config: AgentConfig;
}

export interface UpdateAgentRequest {
  name?: string;
  description?: string;
  config?: Partial<AgentConfig>;
  isActive?: boolean;
}


