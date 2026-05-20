"use client";

import React from "react";
import type { ContinuousAgent } from "../types";
import { LAYOUT_STYLES } from "../constants/styles";
import { AgentCard } from "./AgentCard";
import { AgentErrorBoundary } from "./error-boundary";

type AgentListProps = {
  readonly agents: readonly ContinuousAgent[];
  readonly onToggle: (agentId: string, isActive: boolean) => Promise<void>;
  readonly onDelete: (agentId: string) => Promise<void>;
  readonly onUpdate?: (agent: ContinuousAgent) => Promise<void>;
  readonly onRefresh: () => void;
};

/**
 * List component for displaying agent cards
 */
export const AgentList = ({
  agents,
  onToggle,
  onDelete,
  onUpdate,
  onRefresh,
}: AgentListProps): JSX.Element => {
  return (
    <div
      className={LAYOUT_STYLES.GRID_CARDS}
      role="list"
      aria-label="Lista de agentes continuos"
    >
      {agents.map((agent) => (
        <AgentErrorBoundary key={agent.id}>
          <AgentCard
            agent={agent}
            onToggle={onToggle}
            onDelete={onDelete}
            onUpdate={onUpdate}
            onRefresh={onRefresh}
          />
        </AgentErrorBoundary>
      ))}
    </div>
  );
};

