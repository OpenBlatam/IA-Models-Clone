"use client";

import { useMemo } from "react";
import type { ContinuousAgent } from "../types";
import { getCreditsStatusClass } from "../utils/formatting";

const LOW_CREDITS_THRESHOLD = 1000;

type UseAgentDashboardStatsOptions = {
  readonly agents: readonly ContinuousAgent[];
};

type UseAgentDashboardStatsReturn = {
  readonly activeAgents: readonly ContinuousAgent[];
  readonly totalExecutions: number;
  readonly totalCreditsUsed: number;
  readonly totalCreditsRemaining: number;
  readonly totalSuccessfulExecutions: number;
  readonly successRate: number;
  readonly creditsRemainingClass: string;
  readonly hasLowCredits: boolean;
};

/**
 * Custom hook for calculating agent dashboard statistics
 * 
 * Features:
 * - Memoized calculations for performance
 * - Active agents filtering
 * - Aggregate statistics (executions, credits)
 * - Success rate calculation
 * - Credits status classification
 */
export const useAgentDashboardStats = ({
  agents,
}: UseAgentDashboardStatsOptions): UseAgentDashboardStatsReturn => {
  const activeAgents = useMemo(
    () => agents.filter((agent) => agent.isActive),
    [agents]
  );

  const totalExecutions = useMemo(
    () => agents.reduce((sum, agent) => sum + agent.stats.totalExecutions, 0),
    [agents]
  );

  const totalCreditsUsed = useMemo(
    () => agents.reduce((sum, agent) => sum + agent.stats.creditsUsed, 0),
    [agents]
  );

  const totalCreditsRemaining = useMemo(
    () =>
      agents.reduce(
        (sum, agent) => sum + (agent.stripeCreditsRemaining || 0),
        0
      ),
    [agents]
  );

  const totalSuccessfulExecutions = useMemo(
    () => agents.reduce((sum, agent) => sum + agent.stats.successfulExecutions, 0),
    [agents]
  );

  const successRate = useMemo(() => {
    if (totalExecutions === 0) return 0;
    return (totalSuccessfulExecutions / totalExecutions) * 100;
  }, [totalExecutions, totalSuccessfulExecutions]);

  const hasLowCredits = useMemo(
    () => totalCreditsRemaining < LOW_CREDITS_THRESHOLD,
    [totalCreditsRemaining]
  );

  const creditsRemainingClass = useMemo(
    () =>
      getCreditsStatusClass(
        hasLowCredits ? totalCreditsRemaining : null
      ),
    [hasLowCredits, totalCreditsRemaining]
  );

  return {
    activeAgents,
    totalExecutions,
    totalCreditsUsed,
    totalCreditsRemaining,
    totalSuccessfulExecutions,
    successRate,
    creditsRemainingClass,
    hasLowCredits,
  };
};



