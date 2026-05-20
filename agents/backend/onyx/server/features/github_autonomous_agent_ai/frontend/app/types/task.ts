import { GitHubRepository } from '../lib/github-api';

export type TaskStatus = 'pending' | 'processing' | 'running' | 'completed' | 'failed' | 'stopped' | 'pending_approval' | 'pending_commit_approval';

export interface TaskResult {
  content: string;
  plan?: {
    steps?: string[];
    files_to_create?: string[];
    files_to_modify?: string[];
  };
  code?: string;
}

export interface Task {
  id: string;
  repository: string;
  instruction: string;
  status: TaskStatus;
  createdAt: string;
  processingStartedAt?: string;
  repoInfo?: GitHubRepository;
  model?: string; // Modelo de IA usado
  result?: TaskResult;
  streamingContent?: string;
  error?: string;
  executionStatus?: 'pending' | 'executing' | 'completed' | 'failed';
  executionResult?: {
    success: boolean;
    commitSha?: string;
    branch?: string;
    commitUrl?: string;
    error?: string;
  };
  pendingApproval?: {
    plan: any;
    commitMessage: string;
    actions: Array<{
      path: string;
      content: string;
      action: 'create' | 'update';
    }>;
    approved?: boolean; // Marca si el plan fue aprobado pero no ejecutado todavía
  };
  pendingCommitApproval?: {
    commitSha?: string;
    commitUrl?: string;
    commitMessage: string;
    branch?: string;
  };
}

export const TASK_STORAGE_KEY = 'github_agent_tasks';



