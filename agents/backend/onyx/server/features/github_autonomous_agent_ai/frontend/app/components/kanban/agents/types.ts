export interface ContinuousAgent {
  id: string;
  name: string;
  description?: string;
  isActive: boolean;
  config: {
    taskType: string;
  };
  stats: {
    totalExecutions: number;
    successfulExecutions: number;
    failedExecutions: number;
    creditsUsed: number;
    lastExecutionAt?: string | null;
  };
  createdAt?: string;
}

export interface AgentsSectionProps {
  className?: string;
}

export type FilterStatus = "all" | "active" | "inactive";
export type ViewMode = "cards" | "table";

export interface AgentStats {
  totalExecutions: number;
  totalSuccess: number;
  globalSuccessRate: number;
}








