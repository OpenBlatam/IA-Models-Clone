"use client";

import { useState, useCallback } from "react";
import { toast } from "sonner";
import type { ContinuousAgent } from "../types";
import { useModalState } from "./useModalState";
import { useContinuousAgent } from "./useContinuousAgent";
import { REFRESH_INTERVALS } from "../constants";

type UseAgentCardActionsOptions = {
  readonly initialAgent: ContinuousAgent;
  readonly onToggle: (agentId: string, isActive: boolean) => Promise<void>;
  readonly onDelete: (agentId: string) => Promise<void>;
  readonly onUpdate?: (agent: ContinuousAgent) => Promise<void>;
  readonly onRefresh: () => void;
};

type UseAgentCardActionsReturn = {
  readonly currentAgent: ContinuousAgent;
  readonly isToggling: boolean;
  readonly isDeleting: boolean;
  readonly isEditModalOpen: boolean;
  readonly showDeleteConfirm: boolean;
  readonly handleToggle: () => Promise<void>;
  readonly handleDeleteClick: () => void;
  readonly handleDeleteConfirm: () => Promise<void>;
  readonly handleEditClick: () => void;
  readonly handleUpdate: (agent: ContinuousAgent) => Promise<void>;
  readonly handleCloseEditModal: () => void;
  readonly setShowDeleteConfirm: (show: boolean) => void;
};

/**
 * Custom hook for managing agent card actions (toggle, delete, edit)
 * 
 * Features:
 * - Real-time agent updates via useContinuousAgent
 * - Toggle agent with toast notifications
 * - Delete with confirmation dialog
 * - Edit modal management
 * - Error handling with user-friendly messages
 */
export const useAgentCardActions = ({
  initialAgent,
  onToggle,
  onDelete,
  onUpdate,
  onRefresh,
}: UseAgentCardActionsOptions): UseAgentCardActionsReturn => {
  const [isDeleting, setIsDeleting] = useState(false);
  const editModal = useModalState(false);
  const deleteConfirm = useModalState(false);

  const {
    agent,
    isToggling,
    toggleActive,
  } = useContinuousAgent({
    agentId: initialAgent.id,
    autoRefresh: true,
    refreshInterval: REFRESH_INTERVALS.SINGLE_AGENT,
    onUpdate: (updatedAgent) => {
      if (updatedAgent.isActive !== initialAgent.isActive) {
        onToggle(updatedAgent.id, updatedAgent.isActive);
      }
    },
  });

  const currentAgent = agent || initialAgent;

  const handleToggle = useCallback(async (): Promise<void> => {
    try {
      await toggleActive();
      toast.success(
        `Agente ${currentAgent.isActive ? "desactivado" : "activado"} exitosamente`
      );
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Error al cambiar el estado del agente";
      toast.error(errorMessage);
    }
  }, [toggleActive, currentAgent.isActive]);

  const handleDeleteClick = useCallback((): void => {
    deleteConfirm.open();
  }, [deleteConfirm]);

  const handleDeleteConfirm = useCallback(async (): Promise<void> => {
    if (!currentAgent?.id) {
      return;
    }

    setIsDeleting(true);
    try {
      await onDelete(currentAgent.id);
      toast.success("Agente eliminado exitosamente");
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Error al eliminar el agente";
      toast.error(errorMessage);
      setIsDeleting(false);
    }
  }, [currentAgent.id, onDelete]);

  const handleEditClick = useCallback((): void => {
    editModal.open();
  }, [editModal]);

  const handleUpdate = useCallback(
    async (updatedAgent: ContinuousAgent): Promise<void> => {
      if (onUpdate) {
        await onUpdate(updatedAgent);
      }
      editModal.close();
      onRefresh();
    },
    [onUpdate, editModal, onRefresh]
  );

  return {
    currentAgent,
    isToggling,
    isDeleting,
    isEditModalOpen: editModal.isOpen,
    showDeleteConfirm: deleteConfirm.isOpen,
    handleToggle,
    handleDeleteClick,
    handleDeleteConfirm,
    handleEditClick,
    handleUpdate,
    handleCloseEditModal: editModal.close,
    setShowDeleteConfirm: deleteConfirm.setIsOpen,
  };
};

