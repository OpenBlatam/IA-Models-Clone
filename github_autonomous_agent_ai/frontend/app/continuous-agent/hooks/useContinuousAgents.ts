import { useCallback, useMemo } from "react";
import useSWR from "swr";
import type {
  ContinuousAgent,
  CreateAgentRequest,
  UpdateAgentRequest,
} from "../types";
import { fetchAgents, createAgent, updateAgent, deleteAgent } from "../services/agentService";
import { REFRESH_INTERVALS } from "../constants";
import { getApiErrorMessage } from "../utils/apiError";
import { ERROR_MESSAGES } from "../constants/messages";

/**
 * Options for configuring the useContinuousAgents hook
 */
type UseContinuousAgentsOptions = {
  /** Whether to automatically refresh the agents list */
  readonly autoRefresh?: boolean;
  /** Interval in milliseconds for auto-refresh */
  readonly refreshInterval?: number;
};

/**
 * Return type for useContinuousAgents hook
 */
type UseContinuousAgentsReturn = {
  /** List of continuous agents */
  readonly agents: ContinuousAgent[];
  /** Whether the agents are currently loading */
  readonly isLoading: boolean;
  /** Error message if any */
  readonly error: string | null;
  /** Function to create a new agent */
  readonly createAgent: (request: CreateAgentRequest) => Promise<ContinuousAgent>;
  /** Function to update an existing agent */
  readonly updateAgent: (agentId: string, updates: UpdateAgentRequest) => Promise<ContinuousAgent>;
  /** Function to delete an agent */
  readonly deleteAgent: (agentId: string) => Promise<void>;
  /** Function to toggle agent active state */
  readonly toggleAgent: (agentId: string) => Promise<void>;
  /** Function to manually refresh the agents list */
  readonly refresh: () => Promise<void>;
};

const DEFAULT_AUTO_REFRESH = true;
const AGENTS_API_URL = "/api/continuous-agent";

/**
 * Fetcher function for SWR to fetch agents
 * @returns Promise resolving to array of agents
 * @throws Error if fetch fails
 */
const agentsFetcher = async (): Promise<ContinuousAgent[]> => {
  return fetchAgents();
};

/**
 * Custom hook for managing continuous agents with CRUD operations
 * 
 * Features:
 * - Auto-refresh with configurable interval
 * - Optimistic updates for better UX
 * - Error handling with user-friendly messages
 * - Type-safe operations
 * 
 * @param options - Configuration options for the hook
 * @returns Object with agents data and CRUD operations
 */
export const useContinuousAgents = (
  options: UseContinuousAgentsOptions = {}
): UseContinuousAgentsReturn => {
  const {
    autoRefresh = DEFAULT_AUTO_REFRESH,
    refreshInterval = REFRESH_INTERVALS.AGENTS_LIST,
  } = options;

  const { data, error, isLoading, mutate } = useSWR<ContinuousAgent[]>(
    AGENTS_API_URL,
    agentsFetcher,
    {
      refreshInterval: autoRefresh ? refreshInterval : 0,
      revalidateOnFocus: false,
      revalidateOnReconnect: true,
      dedupingInterval: 2000, // Prevent duplicate requests within 2 seconds
    }
  );

  const agents = useMemo(() => data ?? [], [data]);
  
  const errorMessage = useMemo(
    () => (error ? getApiErrorMessage(error, ERROR_MESSAGES.LOAD_AGENTS) : null),
    [error]
  );

  const createAgentHandler = useCallback(
    async (request: CreateAgentRequest): Promise<ContinuousAgent> => {
      if (!request?.name?.trim() || !request?.description?.trim()) {
        throw new Error("Name and description are required");
      }

      const newAgent = await createAgent(request);
      
      // Optimistic update
      await mutate(
        (currentAgents) => (currentAgents ? [...currentAgents, newAgent] : [newAgent]),
        false
      );
      
      return newAgent;
    },
    [mutate]
  );

  const updateAgentHandler = useCallback(
    async (agentId: string, updates: UpdateAgentRequest): Promise<ContinuousAgent> => {
      if (!agentId?.trim()) {
        throw new Error("Agent ID is required");
      }

      const updatedAgent = await updateAgent(agentId, updates);
      
      // Optimistic update
      await mutate(
        (currentAgents) =>
          currentAgents
            ? currentAgents.map((a) => (a.id === agentId ? updatedAgent : a))
            : null,
        false
      );
      
      return updatedAgent;
    },
    [mutate]
  );

  const deleteAgentHandler = useCallback(
    async (agentId: string): Promise<void> => {
      if (!agentId?.trim()) {
        throw new Error("Agent ID is required");
      }

      await deleteAgent(agentId);
      
      // Optimistic update
      await mutate(
        (currentAgents) =>
          currentAgents ? currentAgents.filter((a) => a.id !== agentId) : null,
        false
      );
    },
    [mutate]
  );

  const toggleAgent = useCallback(
    async (agentId: string): Promise<void> => {
      if (!agentId?.trim()) {
        throw new Error("Agent ID is required");
      }

      const agent = agents.find((a) => a.id === agentId);
      if (!agent) {
        throw new Error(`Agent with ID ${agentId} not found`);
      }

      const previousActiveState = agent.isActive;
      const newActiveState = !previousActiveState;

      // Optimistic update
      await mutate(
        (currentAgents) =>
          currentAgents
            ? currentAgents.map((a) =>
                a.id === agentId ? { ...a, isActive: newActiveState } : a
              )
            : null,
        false
      );

      try {
        await updateAgentHandler(agentId, { isActive: newActiveState });
      } catch (err) {
        // Rollback on error
        await mutate(
          (currentAgents) =>
            currentAgents
              ? currentAgents.map((a) =>
                  a.id === agentId ? { ...a, isActive: previousActiveState } : a
                )
              : null,
          false
        );
        throw err;
      }
    },
    [agents, updateAgentHandler, mutate]
  );

  const refresh = useCallback(async (): Promise<void> => {
    await mutate();
  }, [mutate]);

  return {
    agents,
    isLoading,
    error: errorMessage,
    createAgent: createAgentHandler,
    updateAgent: updateAgentHandler,
    deleteAgent: deleteAgentHandler,
    toggleAgent,
    refresh,
  };
};

