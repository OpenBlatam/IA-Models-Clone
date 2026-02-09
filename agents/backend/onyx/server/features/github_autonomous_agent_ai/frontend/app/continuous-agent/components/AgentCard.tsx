"use client";

import React, { memo } from "react";
import { motion } from "framer-motion";
import { fadeInAnimation } from "../constants/animations";
import { CARD_STYLES, TEXT_STYLES } from "../constants/styles";
import { compareAgentCardProps } from "../utils/memo-comparison";
import type { ContinuousAgent } from "../types";
import { ToggleSwitch } from "./ToggleSwitch";
import { useAgentCardActions } from "../hooks/useAgentCardActions";
import { AgentStats, AgentCredits, AgentGoal } from "./agent";
import { Button } from "./ui/Button";
import { ConfirmDialog } from "./ui/ConfirmDialog";
import { UI_MESSAGES } from "../constants/messages";
import { EditAgentModal } from "./EditAgentModal";

type AgentCardProps = {
  readonly agent: ContinuousAgent;
  readonly onToggle: (agentId: string, isActive: boolean) => void;
  readonly onDelete: (agentId: string) => void;
  readonly onUpdate?: (agent: ContinuousAgent) => Promise<void> | void;
  readonly onRefresh: () => void;
};

/**
 * AgentCard component displays information about a single continuous agent
 * 
 * Features:
 * - Real-time status updates via useContinuousAgent hook
 * - Toggle active/inactive state
 * - Delete agent with confirmation
 * - Displays stats and credits information
 * 
 * @param props - Component props
 * @returns The rendered agent card component
 */
const AgentCardComponent = ({
  agent: initialAgent,
  onToggle,
  onDelete,
  onUpdate,
  onRefresh,
}: AgentCardProps): JSX.Element => {
  const {
    currentAgent,
    isToggling,
    isDeleting,
    isEditModalOpen,
    showDeleteConfirm,
    handleToggle,
    handleDeleteClick,
    handleDeleteConfirm,
    handleEditClick,
    handleUpdate,
    handleCloseEditModal,
    setShowDeleteConfirm,
  } = useAgentCardActions({
    initialAgent,
    onToggle,
    onDelete,
    onUpdate,
    onRefresh,
  });

  return (
    <>
      <motion.article
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.2 }}
        whileHover={{ scale: 1.02 }}
        className={CARD_STYLES.BASE}
      >
      <div className={CARD_STYLES.HEADER}>
        <div className="flex-1">
          <h3 className={TEXT_STYLES.TITLE}>{currentAgent.name}</h3>
          <p className={TEXT_STYLES.SUBTITLE}>{currentAgent.description}</p>
        </div>
        <div className="ml-4">
          <ToggleSwitch
            checked={currentAgent.isActive}
            onChange={handleToggle}
            disabled={isToggling}
            size="md"
            aria-label={`${currentAgent.name} - ${currentAgent.isActive ? "Activo" : "Inactivo"}`}
          />
        </div>
      </div>

      <div className={CARD_STYLES.CONTENT}>
        <AgentStats stats={currentAgent.stats} isActive={currentAgent.isActive} />
        <AgentCredits creditsRemaining={currentAgent.stripeCreditsRemaining} />
        {currentAgent.config.goal && (
          <AgentGoal goal={currentAgent.config.goal} />
        )}
      </div>

      <div className={CARD_STYLES.ACTIONS}>
        {onUpdate && (
          <Button
            type="button"
            variant="default"
            size="sm"
            onClick={handleEditClick}
            ariaLabel={`Editar agente ${currentAgent.name}`}
          >
            Editar
          </Button>
        )}
        <Button
          type="button"
          variant="danger"
          size="sm"
          fullWidth={!onUpdate}
          onClick={handleDeleteClick}
          loading={isDeleting}
          disabled={isDeleting}
          ariaLabel={`Eliminar agente ${currentAgent.name}`}
        >
          {UI_MESSAGES.DELETE}
        </Button>
      </div>

      {onUpdate && (
        <EditAgentModal
          open={isEditModalOpen}
          agent={currentAgent}
          onClose={handleCloseEditModal}
          onUpdate={handleUpdate}
        />
      )}

      {isToggling && (
        <motion.div
          {...fadeInAnimation}
          className="mt-2 text-xs text-muted-foreground text-center"
          role="status"
          aria-live="polite"
        >
          {UI_MESSAGES.CHANGING_STATE}
        </motion.div>
      )}
      </motion.article>

      <ConfirmDialog
        open={showDeleteConfirm}
        onOpenChange={setShowDeleteConfirm}
        title="Confirmar eliminación"
        description={`¿Estás seguro de que deseas eliminar el agente "${currentAgent.name}"? Esta acción no se puede deshacer.`}
        confirmText="Eliminar"
        cancelText="Cancelar"
        variant="danger"
        onConfirm={handleDeleteConfirm}
      />
    </>
  );
};

// Memoize component to prevent unnecessary re-renders
export const AgentCard = memo(AgentCardComponent, (prevProps, nextProps) => {
  return compareAgentCardProps(prevProps, nextProps);
});

AgentCard.displayName = "AgentCard";
