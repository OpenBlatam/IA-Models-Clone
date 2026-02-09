"use client";

import { memo, useMemo } from "react";
import { StatusBadge, ToggleButton } from "./ui";
import { calculateSuccessRate } from "../utils/calculations";
import { formatSuccessRate, formatCredits } from "../utils/formatters";
import { UI_CLASSES } from "../config/uiConstants";
import type { ContinuousAgent } from "../types";

interface AgentTableProps {
  agents: ContinuousAgent[];
  onToggleActive: (agent: ContinuousAgent) => void;
}

const TABLE_HEADERS = [
  "Nombre",
  "Estado",
  "Tipo",
  "Ejecuciones",
  "Éxito",
  "Créditos",
  "Acciones",
] as const;

function AgentTableContent({ agents, onToggleActive }: AgentTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className={UI_CLASSES.table.header}>
          <tr>
            {TABLE_HEADERS.map((header) => (
              <th key={header} className={`${UI_CLASSES.table.cell} text-left text-xs font-semibold text-gray-700`}>
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200">
          {agents.map((agent) => (
            <AgentTableRow key={agent.id} agent={agent} onToggleActive={onToggleActive} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AgentTableRow({
  agent,
  onToggleActive,
}: {
  agent: ContinuousAgent;
  onToggleActive: (agent: ContinuousAgent) => void;
}) {
  const successRate = useMemo(
    () => calculateSuccessRate(agent.stats.successfulExecutions, agent.stats.totalExecutions),
    [agent.stats.successfulExecutions, agent.stats.totalExecutions]
  );

  return (
    <tr className={UI_CLASSES.table.row}>
      <td className={UI_CLASSES.table.cell}>
        <div className="font-medium text-gray-900">{agent.name}</div>
        {agent.description && (
          <div className="text-xs text-gray-500 truncate max-w-xs">{agent.description}</div>
        )}
      </td>
      <td className={UI_CLASSES.table.cell}>
        <StatusBadge isActive={agent.isActive} />
      </td>
      <td className={`${UI_CLASSES.table.cell} text-xs text-gray-600`}>{agent.config.taskType}</td>
      <td className={`${UI_CLASSES.table.cell} text-xs text-gray-600`}>
        {agent.stats.totalExecutions}
      </td>
      <td className={`${UI_CLASSES.table.cell} text-xs text-gray-600`}>
        {agent.stats.totalExecutions ? formatSuccessRate(successRate) : "-"}
      </td>
      <td className={`${UI_CLASSES.table.cell} text-xs text-gray-600`}>
        {formatCredits(agent.stats.creditsUsed)}
      </td>
      <td className={UI_CLASSES.table.cell}>
        <ToggleButton
          isActive={agent.isActive}
          onClick={() => onToggleActive(agent)}
          size="sm"
        />
      </td>
    </tr>
  );
}

export const AgentTable = memo(AgentTableContent);
