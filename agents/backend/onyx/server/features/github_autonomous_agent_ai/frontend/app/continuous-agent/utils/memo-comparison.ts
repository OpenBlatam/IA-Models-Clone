import type { ContinuousAgent } from "../types";

/**
 * Comparison function for AgentCard memo
 * 
 * Only re-renders if agent data or callbacks change
 */
export const compareAgentCardProps = (
  prevProps: { agent: ContinuousAgent; onToggle: unknown; onDelete: unknown; onUpdate?: unknown; onRefresh: unknown },
  nextProps: { agent: ContinuousAgent; onToggle: unknown; onDelete: unknown; onUpdate?: unknown; onRefresh: unknown }
): boolean => {
  return (
    prevProps.agent.id === nextProps.agent.id &&
    prevProps.agent.isActive === nextProps.agent.isActive &&
    prevProps.agent.stats.totalExecutions === nextProps.agent.stats.totalExecutions &&
    prevProps.agent.stripeCreditsRemaining === nextProps.agent.stripeCreditsRemaining &&
    prevProps.agent.config.goal === nextProps.agent.config.goal &&
    prevProps.onToggle === nextProps.onToggle &&
    prevProps.onDelete === nextProps.onDelete &&
    prevProps.onUpdate === nextProps.onUpdate &&
    prevProps.onRefresh === nextProps.onRefresh
  );
};

/**
 * Comparison function for AgentDashboard memo
 * 
 * Only re-renders if agents array or stats change
 */
export const compareAgentDashboardProps = (
  prevProps: { agents: ContinuousAgent[] },
  nextProps: { agents: ContinuousAgent[] }
): boolean => {
  // Only re-render if agents array length changes
  if (prevProps.agents.length !== nextProps.agents.length) {
    return false;
  }

  // Deep comparison of agent stats
  return prevProps.agents.every((prevAgent, index) => {
    const nextAgent = nextProps.agents[index];
    if (!nextAgent || prevAgent.id !== nextAgent.id) {
      return false;
    }
    return (
      prevAgent.isActive === nextAgent.isActive &&
      prevAgent.stats.totalExecutions === nextAgent.stats.totalExecutions &&
      prevAgent.stats.creditsUsed === nextAgent.stats.creditsUsed &&
      prevAgent.stripeCreditsRemaining === nextAgent.stripeCreditsRemaining
    );
  });
};



