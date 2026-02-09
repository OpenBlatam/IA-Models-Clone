import { useState, useCallback, useEffect, useRef, useMemo } from "react";
import useSWR from "swr";
import type { ContinuousAgent, UpdateAgentRequest } from "../types";
import { fetchAgent, updateAgent } from "../services/agentService";
import { REFRESH_INTERVALS } from "../constants";
import { getApiErrorMessage } from "../utils/apiError";
import { ERROR_MESSAGES } from "../constants/messages";

/**
 * Options for configuring the useContinuousAgent hook
 */
type UseContinuousAgentOptions = {
  /** The ID of the agent to manage */
  readonly agentId: string;
  /** Whether to automatically refresh the agent data */
  readonly autoRefresh?: boolean;
  /** Interval in milliseconds for auto-refresh */
  readonly refreshInterval?: number;
  /** Callback invoked when agent data is updated */
  readonly onUpdate?: (agent: ContinuousAgent) => void;
};

/**
 * Return type for useContinuousAgent hook
 */
type UseContinuousAgentReturn = {
  /** The agent data or null if not loaded */
  readonly agent: ContinuousAgent | null;
  /** Whether the agent is currently loading */
  readonly isLoading: boolean;
  /** Error message if any */
  readonly error: string | null;
  /** Whether a toggle operation is in progress */
  readonly isToggling: boolean;
  /** Function to toggle agent active state */
  readonly toggleActive: () => Promise<void>;
  /** Function to update agent data */
  readonly updateAgent: (updates: UpdateAgentRequest) => Promise<void>;
  /** Function to manually refresh agent data */
  readonly refresh: () => Promise<void>;
};

const DEFAULT_AUTO_REFRESH = true;

/**
 * Fetcher function for SWR to fetch a single agent
 * @param agentId - The ID of the agent to fetch
 * @returns Promise resolving to the agent
 * @throws Error if fetch fails
 */
const agentFetcher = async (agentId: string): Promise<ContinuousAgent> => {
  if (!agentId?.trim()) {
    throw new Error("Agent ID is required");
  }
  return fetchAgent(agentId);
};

/**
 * Custom hook for managing a single continuous agent with real-time updates
 * 
 * Features:
 * - Auto-refresh with configurable interval
 * - Optimistic updates for toggle operations
 * - Error handling with user-friendly messages
 * - Type-safe operations
 * 
 * @param options - Configuration options for the hook
 * @returns Object with agent data and operations
 */
export const useContinuousAgent = (
  options: UseContinuousAgentOptions
): UseContinuousAgentReturn => {
  const {
    agentId,
    autoRefresh = DEFAULT_AUTO_REFRESH,
    refreshInterval = REFRESH_INTERVALS.SINGLE_AGENT,
    onUpdate,
  } = options;

  // Early return for invalid agentId
  const isValidAgentId = agentId?.trim();
  
  if (!isValidAgentId) {
    const errorState = {
      agent: null as ContinuousAgent | null,
      isLoading: false,
      error: "Agent ID is required" as string | null,
      isToggling: false,
      toggleActive: async (): Promise<void> => {
        throw new Error("Agent ID is required");
      },
      updateAgent: async (): Promise<void> => {
        throw new Error("Agent ID is required");
      },
      refresh: async (): Promise<void> => {
        throw new Error("Agent ID is required");
      },
    };
    return errorState;
  }

  const [isToggling, setIsToggling] = useState(false);
  const onUpdateRef = useRef(onUpdate);

  // Keep callback ref up to date
  useEffect(() => {
    onUpdateRef.current = onUpdate;
  }, [onUpdate]);

  // Memoize SWR key to prevent unnecessary re-fetches
  const swrKey = useMemo(
    () => [`/api/continuous-agent/${isValidAgentId}`, isValidAgentId],
    [isValidAgentId]
  );

  const { data, error, isLoading, mutate } = useSWR<ContinuousAgent>(
    swrKey,
    () => agentFetcher(isValidAgentId),
    {
      refreshInterval: autoRefresh ? refreshInterval : 0,
      revalidateOnFocus: false,
      revalidateOnReconnect: true,
      dedupingInterval: 2000, // Prevent duplicate requests within 2 seconds
    }
  );

  const agent = useMemo(() => data ?? null, [data]);
  
  const errorMessage = useMemo(
    () => (error ? getApiErrorMessage(error, ERROR_MESSAGES.LOAD_AGENT) : null),
    [error]
  );

  // Notify parent component of updates
  useEffect(() => {
    if (agent && onUpdateRef.current) {
      onUpdateRef.current(agent);
    }
  }, [agent]);

  /**
   * Toggles the active state of the agent with optimistic updates
   */
  const toggleActive = useCallback(async (): Promise<void> => {
    if (!agent) {
      throw new Error("Agent not loaded");
    }

    if (isToggling) {
      return; // Prevent concurrent toggles
    }

    const newActiveState = !agent.isActive;
    const previousActiveState = agent.isActive;
    setIsToggling(true);

    // Optimistic update
    await mutate(
      (currentAgent) =>
        currentAgent ? { ...currentAgent, isActive: newActiveState } : null,
      false
    );

    try {
      const updatedAgent = await updateAgent(isValidAgentId, { isActive: newActiveState });
      await mutate(updatedAgent, false);
      onUpdateRef.current?.(updatedAgent);
    } catch (err) {
      // Rollback on error
      await mutate(
        (currentAgent) =>
          currentAgent ? { ...currentAgent, isActive: previousActiveState } : null,
        false
      );
      throw err;
    } finally {
      setIsToggling(false);
    }
  }, [agent, isValidAgentId, isToggling, mutate]);

  /**
   * Updates agent data
   */
  const updateAgentData = useCallback(
    async (updates: UpdateAgentRequest): Promise<void> => {
      if (!agent) {
        throw new Error("Agent not loaded");
      }

      if (!updates || Object.keys(updates).length === 0) {
        throw new Error("Update request must contain at least one field");
      }

      const updatedAgent = await updateAgent(isValidAgentId, updates);
      await mutate(updatedAgent, false);
      onUpdateRef.current?.(updatedAgent);
    },
    [agent, isValidAgentId, mutate]
  );

  const refresh = useCallback(async (): Promise<void> => {
    await mutate();
  }, [mutate]);

  return {
    agent,
    isLoading,
    error: errorMessage,
    isToggling,
    toggleActive,
    updateAgent: updateAgentData,
    refresh,
  };
};

