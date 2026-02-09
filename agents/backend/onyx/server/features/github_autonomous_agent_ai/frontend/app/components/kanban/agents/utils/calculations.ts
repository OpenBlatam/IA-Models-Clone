export function calculateSuccessRate(successful: number, total: number): number {
  if (!total) return 0;
  return (successful * 100) / total;
}

export function calculateAgentStats(agents: import("../types").ContinuousAgent[]) {
  const totalExecutions = agents.reduce((acc, a) => acc + a.stats.totalExecutions, 0);
  const totalSuccess = agents.reduce((acc, a) => acc + a.stats.successfulExecutions, 0);
  const globalSuccessRate = calculateSuccessRate(totalSuccess, totalExecutions);

  return {
    totalExecutions,
    totalSuccess,
    globalSuccessRate,
  };
}








