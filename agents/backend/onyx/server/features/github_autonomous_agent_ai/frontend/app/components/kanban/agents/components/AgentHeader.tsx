"use client";

import { StatsDisplay } from "./ui";
import { ExpandButton } from "./ui/ExpandButton";
import { ViewModeToggle } from "./ui/ViewModeToggle";
import { formatSuccessRate } from "../utils/formatters";
import type { AgentStats, ViewMode } from "../types";

interface AgentHeaderProps {
  agentCount: number;
  stats: AgentStats;
  isExpanded: boolean;
  viewMode: ViewMode;
  onToggleExpand: () => void;
  onToggleViewMode: () => void;
  className?: string;
}

export function AgentHeader({
  agentCount,
  stats,
  isExpanded,
  viewMode,
  onToggleExpand,
  onToggleViewMode,
  className,
}: AgentHeaderProps) {
  return (
    <div className={`p-4 border-b border-gray-200 flex items-center justify-between gap-4 ${className}`}>
      <div className="flex items-center gap-3">
        <ExpandButton isExpanded={isExpanded} onClick={onToggleExpand} />
        <div>
          <h3 className="text-sm font-semibold text-gray-900">Agentes continuos</h3>
          <p className="text-xs text-gray-500">
            {agentCount} agente{agentCount !== 1 ? "s" : ""} configurado
            {agentCount !== 1 ? "s" : ""}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <div className="hidden sm:flex items-center gap-3 px-3 py-1.5 rounded-lg bg-gray-50 border border-gray-200">
          <StatsDisplay label="Ejecuciones totales" value={stats.totalExecutions} />
          <div className="w-px h-6 bg-gray-200" />
          <StatsDisplay
            label="Tasa de éxito"
            value={formatSuccessRate(stats.globalSuccessRate)}
          />
        </div>
        <ViewModeToggle currentMode={viewMode} onToggle={onToggleViewMode} />
      </div>
    </div>
  );
}
