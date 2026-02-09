import React, { memo, useMemo } from "react";
import type { AgentStats as AgentStatsType } from "../../types";
import { formatRelativeTime } from "../../utils/dateUtils";
import { formatAgentStatus, getStatusClass } from "../../utils/agent-formatting";
import { CARD_STYLES, STATUS_STYLES, TEXT_STYLES } from "../../constants/styles";
import { cn } from "../../utils/classNames";

type AgentStatsProps = {
  readonly stats: AgentStatsType;
  readonly isActive: boolean;
};

/**
 * Component displaying agent execution statistics
 * 
 * Features:
 * - Real-time status display
 * - Formatted relative times
 * - Success execution count
 * - Accessible labels
 * 
 * @param props - Component props
 * @returns The rendered stats component
 */
const AgentStatsComponent = ({ stats, isActive }: AgentStatsProps): JSX.Element => {
  // Memoize formatted times to prevent unnecessary recalculations
  const lastExecution = useMemo(
    () => formatRelativeTime(stats.lastExecutionAt),
    [stats.lastExecutionAt]
  );
  const nextExecution = useMemo(
    () => formatRelativeTime(stats.nextExecutionAt),
    [stats.nextExecutionAt]
  );

  // Memoize computed values
  const statusText = useMemo(() => formatAgentStatus(isActive), [isActive]);
  const statusClass = useMemo(() => getStatusClass(isActive), [isActive]);
  const nextExecutionClass = useMemo(
    () => (isActive ? STATUS_STYLES.NEXT_EXECUTION_ACTIVE : ""),
    [isActive]
  );

  return (
    <>
      <div className={CARD_STYLES.STATS_ROW}>
        <span className={TEXT_STYLES.LABEL}>Estado:</span>
        <span className={cn(TEXT_STYLES.VALUE, statusClass)} aria-live="polite">
          {statusText}
        </span>
      </div>
      <div className={CARD_STYLES.STATS_ROW}>
        <span className={TEXT_STYLES.LABEL}>Última ejecución:</span>
        <span aria-label={`Última ejecución ${lastExecution}`}>{lastExecution}</span>
      </div>
      <div className={CARD_STYLES.STATS_ROW}>
        <span className={TEXT_STYLES.LABEL}>Próxima ejecución:</span>
        <span className={nextExecutionClass} aria-label={`Próxima ejecución ${nextExecution}`}>
          {nextExecution}
        </span>
      </div>
      <div className={CARD_STYLES.STATS_ROW}>
        <span className={TEXT_STYLES.LABEL}>Ejecuciones exitosas:</span>
        <span className={STATUS_STYLES.SUCCESS} aria-live="polite">
          {stats.successfulExecutions}
        </span>
      </div>
    </>
  );
};

// Memoize component to prevent unnecessary re-renders
export const AgentStats = memo(AgentStatsComponent, (prevProps, nextProps) => {
  return (
    prevProps.isActive === nextProps.isActive &&
    prevProps.stats.totalExecutions === nextProps.stats.totalExecutions &&
    prevProps.stats.successfulExecutions === nextProps.stats.successfulExecutions &&
    prevProps.stats.lastExecutionAt === nextProps.stats.lastExecutionAt &&
    prevProps.stats.nextExecutionAt === nextProps.stats.nextExecutionAt
  );
});

AgentStats.displayName = "AgentStats";

