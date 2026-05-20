"use client";

import { useCallback } from "react";
import type { CreateAgentRequest, ContinuousAgent } from "../types";
import { useAgentErrorHandler } from "./useAgentErrorHandler";

type UseAgentHandlersOptions = {
  readonly createAgent: (request: CreateAgentRequest) => Promise<ContinuousAgent>;
  readonly updateAgent: (id: string, request: Partial<CreateAgentRequest>) => Promise<ContinuousAgent>;
  readonly deleteAgent: (id: string) => Promise<void>;
  readonly toggleAgent: (id: string) => Promise<ContinuousAgent>;
  readonly onModalClose?: () => void;
};

type UseAgentHandlersReturn = {
  readonly handleCreateAgent: (request: CreateAgentRequest) => Promise<void>;
  readonly handleUpdateAgent: (agent: ContinuousAgent) => Promise<void>;
  readonly handleDeleteAgent: (agentId: string) => Promise<void>;
  readonly handleToggleAgent: (agentId: string, isActive: boolean) => Promise<void>;
};

/**
 * Custom hook for managing agent CRUD operations with error handling
 * 
 * Features:
 * - Centralized error handling
 * - Consistent error messages
 * - Modal management integration
 */
export const useAgentHandlers = ({
  createAgent,
  updateAgent,
  deleteAgent,
  toggleAgent,
  onModalClose,
}: UseAgentHandlersOptions): UseAgentHandlersReturn => {
  const errorHandler = useAgentErrorHandler();

  const handleCreateAgent = useCallback(
    async (request: CreateAgentRequest): Promise<void> => {
      const result = await errorHandler.handleCreateAgent(async () => {
        return await createAgent(request);
      });

      if (result && onModalClose) {
        onModalClose();
      }
    },
    [createAgent, errorHandler, onModalClose]
  );

  const handleUpdateAgent = useCallback(
    async (agent: ContinuousAgent): Promise<void> => {
      await errorHandler.handleUpdateAgent(async () => {
        await updateAgent(agent.id, {
          name: agent.name,
          description: agent.description,
          config: agent.config,
        });
      });
    },
    [updateAgent, errorHandler]
  );

  const handleDeleteAgent = useCallback(
    async (agentId: string): Promise<void> => {
      if (!agentId) {
        return;
      }

      await errorHandler.handleDeleteAgent(async () => {
        await deleteAgent(agentId);
      });
    },
    [deleteAgent, errorHandler]
  );

  const handleToggleAgent = useCallback(
    async (agentId: string, isActive: boolean): Promise<void> => {
      if (!agentId) {
        return;
      }

      await errorHandler.handleToggleAgent(
        async () => {
          await toggleAgent(agentId);
        },
        isActive
      );
    },
    [toggleAgent, errorHandler]
  );

  return {
    handleCreateAgent,
    handleUpdateAgent,
    handleDeleteAgent,
    handleToggleAgent,
  };
};



