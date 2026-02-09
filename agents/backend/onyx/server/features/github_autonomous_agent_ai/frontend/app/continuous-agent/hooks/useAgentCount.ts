import { useMemo } from "react";
import type { ContinuousAgent } from "../types";
import { getAgentCountText } from "../utils/agent-count";

type UseAgentCountOptions = {
  readonly agents: ContinuousAgent[];
  readonly filteredAgents: ContinuousAgent[];
};

/**
 * Hook for calculating and formatting agent count text
 * 
 * @param options - Configuration options
 * @returns Formatted agent count text
 */
export const useAgentCount = ({ agents, filteredAgents }: UseAgentCountOptions): string => {
  return useMemo(() => {
    if (filteredAgents.length !== agents.length) {
      return `${filteredAgents.length} de ${agents.length} agente${agents.length !== 1 ? "s" : ""}`;
    }
    return getAgentCountText(agents.length);
  }, [agents.length, filteredAgents.length]);
};



