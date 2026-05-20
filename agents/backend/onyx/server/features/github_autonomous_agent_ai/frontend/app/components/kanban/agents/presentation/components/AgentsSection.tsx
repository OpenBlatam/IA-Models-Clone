"use client";

import { motion, AnimatePresence } from "framer-motion";
import { cn } from "../../../../../utils/cn";
import { AgentsErrorBoundary } from "../error/ErrorBoundary";
import { AgentsProvider } from "../context/AgentsContext";
import { useAgentsContainer } from "../hooks/useAgentsContainer";
import { useAgentsSection } from "../hooks/useAgentsSection";
import { AgentHeader } from "../../components/AgentHeader";
import { AgentFilters } from "../../components/AgentFilters";
import { AgentsList } from "./AgentsList";
import { UI_CLASSES } from "../../config/uiConstants";
import type { AgentsSectionProps } from "../../types";

function AgentsSectionContent({ className }: AgentsSectionProps) {
  const {
    isExpanded,
    viewMode,
    isLoading,
    filteredAgents,
    stats,
    agentCount,
    searchQuery,
    setSearchQuery,
    filterStatus,
    setFilterStatus,
    hasActiveFilters,
    clearFilters,
    toggleExpand,
    toggleViewMode,
    handleToggleActive,
  } = useAgentsSection();

  return (
    <div className={cn(UI_CLASSES.container, className)}>
      <AgentHeader
        agentCount={agentCount}
        stats={stats}
        isExpanded={isExpanded}
        viewMode={viewMode}
        onToggleExpand={toggleExpand}
        onToggleViewMode={toggleViewMode}
      />

      <AnimatePresence initial={false}>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
          >
            <AgentFilters
              searchQuery={searchQuery}
              filterStatus={filterStatus}
              onSearchChange={setSearchQuery}
              onFilterStatusChange={setFilterStatus}
              onClearFilters={clearFilters}
            />

            <div className={UI_CLASSES.content}>
              <AgentsList
                isLoading={isLoading}
                agents={filteredAgents}
                viewMode={viewMode}
                hasActiveFilters={hasActiveFilters}
                onToggleActive={handleToggleActive}
                onClearFilters={clearFilters}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function AgentsSection(props: AgentsSectionProps) {
  const agentsContextValue = useAgentsContainer();

  return (
    <AgentsErrorBoundary>
      <AgentsProvider value={agentsContextValue}>
        <AgentsSectionContent {...props} />
      </AgentsProvider>
    </AgentsErrorBoundary>
  );
}
