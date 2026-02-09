import React from "react";
import type { ContinuousAgent } from "../../types";
import { formatNextExecutionDate } from "../../utils/agent-formatting";

export const ActiveAgentsList = ({ agents }: ActiveAgentsListProps): JSX.Element => {
  if (agents.length === 0) {
    return (
      <p className="text-muted-foreground" role="status">
        No hay agentes activos
      </p>
    );
  }

  return (
    <div
      className="space-y-2"
      role="list"
      aria-labelledby="active-agents-title"
    >
      {agents.map((agent) => (
        <div
          key={agent.id}
          className="flex justify-between items-center p-3 bg-muted rounded-lg"
          role="listitem"
          aria-label={`Agente ${agent.name}`}
        >
          <div>
            <div className="font-medium">{agent.name}</div>
            <div className="text-sm text-muted-foreground">
              Próxima ejecución: {formatNextExecutionDate(agent.stats.nextExecutionAt)}
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm font-medium">
              {agent.stats.successfulExecutions} exitosas
            </div>
            <div className="text-xs text-muted-foreground">
              {agent.stats.failedExecutions} fallidas
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};





