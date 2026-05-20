"use client";

import { memo } from "react";
import { AgentCard } from "../../components/AgentCard";
import { AgentTable } from "../../components/AgentTable";
import { EmptyState } from "../../components/EmptyState";
import { LoadingSpinner } from "../../components/ui";
import { SPACING } from "../../config/uiConstants";
import type { ContinuousAgent, ViewMode } from "../../types";

interface AgentsListProps {
  isLoading: boolean;
  agents: ContinuousAgent[];
  viewMode: ViewMode;
  hasActiveFilters: boolean;
  onToggleActive: (agent: ContinuousAgent) => void;
  onClearFilters: () => void;
}

export const AgentsList = memo(function AgentsList({
  isLoading,
  agents,
  viewMode,
  hasActiveFilters,
  onToggleActive,
  onClearFilters,
}: AgentsListProps) {
  if (isLoading) {
    return <LoadingSpinner message="Cargando agentes..." />;
  }

  if (agents.length === 0) {
    return <EmptyState hasFilters={hasActiveFilters} onClearFilters={onClearFilters} />;
  }

  if (viewMode === "table") {
    return <AgentTable agents={agents} onToggleActive={onToggleActive} />;
  }

  return (
    <div className={SPACING.spaceY3}>
      {agents.map((agent) => (
        <AgentCard key={agent.id} agent={agent} onToggleActive={onToggleActive} />
      ))}
    </div>
  );
});





