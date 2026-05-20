"use client";

import { memo, useMemo } from "react";
import { cn } from "../../../../utils/cn";
import { StatusIndicator, ToggleButton } from "./ui";
import { calculateSuccessRate } from "../utils/calculations";
import { formatSuccessRate, formatCredits, formatDate } from "../utils/formatters";
import { UI_CLASSES } from "../config/uiConstants";
import type { ContinuousAgent } from "../types";

interface AgentCardProps {
  agent: ContinuousAgent;
  onToggleActive: (agent: ContinuousAgent) => void;
}

function AgentCardContent({ agent, onToggleActive }: AgentCardProps) {
  const successRate = useMemo(
    () => calculateSuccessRate(agent.stats.successfulExecutions, agent.stats.totalExecutions),
    [agent.stats.successfulExecutions, agent.stats.totalExecutions]
  );

  const cardClasses = useMemo(
    () =>
      cn(
        UI_CLASSES.card.base,
        agent.isActive ? UI_CLASSES.card.active : UI_CLASSES.card.inactive
      ),
    [agent.isActive]
  );

  return (
    <div className={cardClasses}>
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <StatusIndicator isActive={agent.isActive} />
            <h4 className="text-sm font-semibold text-gray-900">{agent.name}</h4>
            <span className="px-1.5 py-0.5 text-[11px] rounded-full bg-gray-100 text-gray-600">
              {agent.config.taskType}
            </span>
          </div>
          {agent.description && (
            <p className="mt-1 text-xs text-gray-600 line-clamp-2">{agent.description}</p>
          )}
          <div className="mt-2 grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
            <StatItem label="Ejecuciones:" value={agent.stats.totalExecutions} />
            <StatItem
              label="Éxito:"
              value={agent.stats.totalExecutions ? formatSuccessRate(successRate) : "-"}
            />
            <StatItem label="Créditos:" value={formatCredits(agent.stats.creditsUsed)} />
            {agent.stats.lastExecutionAt && (
              <StatItem label="Última ejecución:" value={formatDate(agent.stats.lastExecutionAt)} />
            )}
          </div>
        </div>
        <ToggleButton
          isActive={agent.isActive}
          onClick={() => onToggleActive(agent)}
          size="md"
        />
      </div>
    </div>
  );
}

function StatItem({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center gap-1">
      <span className="text-gray-500">{label}</span>
      <span className="font-medium text-gray-900">{value}</span>
    </div>
  );
}

export const AgentCard = memo(AgentCardContent, (prev, next) => {
  return (
    prev.agent.id === next.agent.id &&
    prev.agent.isActive === next.agent.isActive &&
    prev.agent.stats.totalExecutions === next.agent.stats.totalExecutions &&
    prev.agent.stats.successfulExecutions === next.agent.stats.successfulExecutions &&
    prev.agent.stats.creditsUsed === next.agent.stats.creditsUsed &&
    prev.agent.stats.lastExecutionAt === next.agent.stats.lastExecutionAt
  );
});
