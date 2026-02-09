"use client";

import React, { useCallback } from "react";

type AgentPageToolbarProps = {
  readonly agentCountText: string;
  readonly onOpenCreateModal: () => void;
};

/**
 * Toolbar component with agent count and create button
 */
export const AgentPageToolbar = ({
  agentCountText,
  onOpenCreateModal,
}: AgentPageToolbarProps): JSX.Element => {
  const handleCreateButtonKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLButtonElement>): void => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        onOpenCreateModal();
      }
    },
    [onOpenCreateModal]
  );

  return (
    <div className="flex justify-between items-center">
      <div className="text-sm text-muted-foreground" aria-live="polite">
        {agentCountText}
      </div>
      <button
        type="button"
        onClick={onOpenCreateModal}
        onKeyDown={handleCreateButtonKeyDown}
        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
        aria-label="Crear nuevo agente"
        tabIndex={0}
      >
        + Crear Nuevo Agente
      </button>
    </div>
  );
};



