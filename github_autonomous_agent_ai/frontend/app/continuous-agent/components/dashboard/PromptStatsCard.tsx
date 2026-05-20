import React, { useMemo } from "react";
import type { ContinuousAgent } from "../../types";
import { StatCard } from "./StatCard";
import { analyzePrompts, getPromptHealthScore } from "../../utils/prompt-analytics";

type PromptStatsCardProps = {
  readonly agents: ContinuousAgent[];
};

/**
 * Component displaying statistics about agent prompts
 * 
 * Features:
 * - Count of agents with prompts
 * - Count of Perplexity-style prompts
 * - Average prompt length
 * - Validation statistics
 * 
 * @param props - Component props
 * @returns The rendered prompt stats component
 */
export const PromptStatsCard = ({ agents }: PromptStatsCardProps): JSX.Element => {
  const analytics = useMemo(() => analyzePrompts(agents), [agents]);
  const healthScore = useMemo(() => getPromptHealthScore(agents), [agents]);

  if (analytics.agentsWithPrompts === 0) {
    return (
      <div className="bg-muted rounded-lg p-4">
        <div className="text-sm text-muted-foreground mb-1">Estadísticas de Prompts</div>
        <div className="text-2xl font-bold">N/A</div>
        <div className="text-xs text-muted-foreground mt-1">
          Ningún agente tiene prompt configurado
        </div>
      </div>
    );
  }

  const promptPercentage = Math.round(
    (analytics.agentsWithPrompts / analytics.totalAgents) * 100
  );
  const perplexityPercentage = analytics.agentsWithPrompts > 0
    ? Math.round((analytics.perplexityStylePrompts / analytics.agentsWithPrompts) * 100)
    : 0;
  const validPercentage = analytics.agentsWithPrompts > 0
    ? Math.round((analytics.validPrompts / analytics.agentsWithPrompts) * 100)
    : 0;

  const healthColorClass = healthScore >= 80 
    ? "text-green-600" 
    : healthScore >= 60 
    ? "text-yellow-600" 
    : "text-red-600";

  return (
    <div className="bg-muted rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-sm font-semibold text-muted-foreground">
          Estadísticas de Prompts
        </div>
        <div className={`text-sm font-bold ${healthColorClass}`}>
          Salud: {healthScore}%
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="text-lg font-bold">{analytics.agentsWithPrompts}</div>
          <div className="text-xs text-muted-foreground">
            Con prompts ({promptPercentage}%)
          </div>
        </div>
        
        <div>
          <div className="text-lg font-bold text-blue-600">
            {analytics.perplexityStylePrompts}
          </div>
          <div className="text-xs text-muted-foreground">
            Estilo Perplexity ({perplexityPercentage}%)
          </div>
        </div>
        
        <div>
          <div className="text-lg font-bold text-green-600">
            {analytics.validPrompts}
          </div>
          <div className="text-xs text-muted-foreground">
            Válidos ({validPercentage}%)
          </div>
        </div>
        
        <div>
          <div className="text-lg font-bold">
            {analytics.averagePromptLength.toLocaleString()}
          </div>
          <div className="text-xs text-muted-foreground">
            Promedio caracteres
          </div>
        </div>
      </div>

      {analytics.promptsWithErrors > 0 && (
        <div className="pt-2 border-t border-gray-300">
          <div className="text-xs text-red-600">
            ⚠ {analytics.promptsWithErrors} prompt{analytics.promptsWithErrors !== 1 ? "s" : ""} con errores
          </div>
        </div>
      )}
    </div>
  );
};

