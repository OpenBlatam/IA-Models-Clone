import { useMemo } from "react";
import { calculateAgentStats } from "../../utils/calculations";
import { container } from "../../infrastructure/container/DIContainer";
import type { AgentEntity } from "../../domain/entities/Agent";

export function useAgentStatsContainer(agents: AgentEntity[]) {
  return useMemo(() => {
    const dtos = container.adapter.toDTOArray(agents);
    return calculateAgentStats(dtos);
  }, [agents]);
}
