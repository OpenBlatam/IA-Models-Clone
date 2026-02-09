"use client";

import React, { memo } from "react";
import type { ContinuousAgent } from "../types";
import { compareAgentDashboardProps } from "../utils/memo-comparison";
import { formatNumber } from "../utils/formatting";
import { StatCard, ActiveAgentsList, PromptStatsCard } from "./dashboard";
import { useAgentDashboardStats } from "../hooks/useAgentDashboardStats";

type AgentDashboardProps = {
  readonly agents: ContinuousAgent[];
};

/**
 * Dashboard component displaying aggregate statistics for all agents
 * 
 * Features:
 * - Active agents count
 * - Total executions and success rate
 * - Credits used and remaining
 * - List of active agents
 * 
 * @param props - Component props
 * @returns The rendered dashboard component
 */
const AgentDashboardComponent = ({ agents }: AgentDashboardProps): JSX.Element => {
  if (!agents || agents.length === 0) {
    return (
      <section className="bg-card border rounded-lg p-6" aria-labelledby="dashboard-title">
        <h2 id="dashboard-title" className="text-2xl font-bold mb-6">
          Dashboard General
        </h2>
        <p className="text-muted-foreground">No hay agentes para mostrar estadísticas.</p>
      </section>
    );
  }

  const {
    activeAgents,
    totalExecutions,
    totalCreditsUsed,
    totalCreditsRemaining,
    totalSuccessfulExecutions,
    creditsRemainingClass,
  } = useAgentDashboardStats({ agents });

  return (
    <section className="bg-card border rounded-lg p-6" aria-labelledby="dashboard-title">
      <h2 id="dashboard-title" className="text-2xl font-bold mb-6">
        Dashboard General
      </h2>

      <div
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6"
        role="list"
        aria-label="Estadísticas generales"
      >
        <StatCard
          label="Agentes Activos"
          value={activeAgents.length}
          subtitle={`de ${agents.length} total`}
          ariaLabel={`${activeAgents.length} agentes activos de ${agents.length} total`}
        />

        <StatCard
          label="Total Ejecuciones"
          value={totalExecutions}
          subtitle={`${formatNumber(totalSuccessfulExecutions)} exitosas`}
          ariaLabel={`${formatNumber(totalExecutions)} ejecuciones totales, ${formatNumber(totalSuccessfulExecutions)} exitosas`}
        />

        <StatCard
          label="Créditos Usados"
          value={totalCreditsUsed}
          subtitle="Total acumulado"
          ariaLabel={`${formatNumber(totalCreditsUsed)} créditos usados en total`}
        />

        <StatCard
          label="Créditos Restantes"
          value={totalCreditsRemaining}
          subtitle="Disponibles en Stripe"
          valueClassName={creditsRemainingClass}
          ariaLabel={`${formatNumber(totalCreditsRemaining)} créditos restantes disponibles en Stripe`}
        />
      </div>

      <div className="mt-6 space-y-6">
        <div>
          <PromptStatsCard agents={agents} />
        </div>
        
        <div>
          <h3 id="active-agents-title" className="text-lg font-semibold mb-4">
            Agentes Activos
          </h3>
          <ActiveAgentsList agents={activeAgents} />
        </div>
      </div>
    </section>
  );
};

// Memoize component to prevent unnecessary re-renders
export const AgentDashboard = memo(AgentDashboardComponent, (prevProps, nextProps) => {
  return compareAgentDashboardProps(prevProps, nextProps);
});

AgentDashboard.displayName = "AgentDashboard";

