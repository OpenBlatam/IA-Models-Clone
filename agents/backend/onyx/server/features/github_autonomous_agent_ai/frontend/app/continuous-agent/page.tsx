"use client";

import React, { useMemo } from "react";
import dynamic from "next/dynamic";
import { AgentErrorBoundary } from "./components/error-boundary";
import { AgentPageHeader } from "./components/AgentPageHeader";
import { AgentPageToolbar } from "./components/AgentPageToolbar";
import { AgentFilters } from "./components/AgentFilters";
import { AgentEmptyState } from "./components/AgentEmptyState";
import { AgentList } from "./components/AgentList";
import { AgentLoadingState } from "./components/AgentLoadingState";
import { AgentErrorState } from "./components/AgentErrorState";
import { useContinuousAgents } from "./hooks/useContinuousAgents";
import { useAgentFilters } from "./hooks/useAgentFilters";
import { useAgentHandlers } from "./hooks/useAgentHandlers";
import { useCreateAgentModal } from "./hooks/useCreateAgentModal";
import { useAgentCount } from "./hooks/useAgentCount";
import { REFRESH_INTERVALS } from "./constants";

// Dynamic imports for code splitting
const AgentDashboard = dynamic(
  () => import("./components/AgentDashboard").then((mod) => ({ default: mod.AgentDashboard })),
  {
    loading: () => <div className="text-center py-8">Cargando dashboard...</div>,
    ssr: false,
  }
);

const CreateAgentModal = dynamic(
  () =>
    import("./components/CreateAgentModal").then((mod) => ({ default: mod.CreateAgentModal })),
  {
    ssr: false,
  }
);

/**
 * Main page component for Continuous Agent management
 * 
 * Features:
 * - Real-time agent list with auto-refresh
 * - Create, update, delete, and toggle agents
 * - Dashboard with statistics
 * - Error handling with user-friendly messages
 * 
 * @returns The rendered page component
 */
const ContinuousAgentPage = (): JSX.Element => {
  const {
    agents,
    isLoading,
    error,
    createAgent,
    updateAgent,
    deleteAgent,
    toggleAgent,
    refresh,
  } = useContinuousAgents({
    autoRefresh: true,
    refreshInterval: REFRESH_INTERVALS.AGENTS_LIST,
  });

  const {
    searchQuery,
    filterStatus,
    sortField,
    sortOrder,
    setSearchQuery,
    setFilterStatus,
    handleSortChange,
    clearFilters,
    filteredAndSortedAgents,
    hasActiveFilters,
  } = useAgentFilters({ agents });

  const createModal = useCreateAgentModal();

  const {
    handleCreateAgent,
    handleUpdateAgent,
    handleDeleteAgent,
    handleToggleAgent,
  } = useAgentHandlers({
    createAgent,
    updateAgent,
    deleteAgent,
    toggleAgent,
    onModalClose: createModal.close,
  });

  const hasAgents = useMemo(() => agents.length > 0, [agents.length]);
  const agentCountText = useAgentCount({
    agents,
    filteredAgents: filteredAndSortedAgents,
  });

  // Early return for loading state
  if (isLoading) {
    return <AgentLoadingState />;
  }

  // Early return for error state
  if (error) {
    return <AgentErrorState error={error} />;
  }

  return (
    <AgentErrorBoundary>
      <div className="min-h-screen bg-background p-8">
        <div className="max-w-7xl mx-auto">
          <AgentPageHeader />

          <div className="mb-6 space-y-4">
            <AgentPageToolbar
              agentCountText={agentCountText}
              onOpenCreateModal={createModal.open}
            />

            {hasAgents && (
              <AgentFilters
                searchQuery={searchQuery}
                onSearchChange={setSearchQuery}
                filterStatus={filterStatus}
                onFilterChange={setFilterStatus}
                sortField={sortField}
                sortOrder={sortOrder}
                onSortChange={handleSortChange}
              />
            )}
          </div>

          {!hasAgents ? (
            <AgentEmptyState
              hasFilters={false}
              onCreateAgent={createModal.open}
            />
          ) : filteredAndSortedAgents.length === 0 ? (
            <AgentEmptyState
              hasFilters={hasActiveFilters}
              onClearFilters={clearFilters}
              onCreateAgent={createModal.open}
            />
          ) : (
            <AgentList
              agents={filteredAndSortedAgents}
              onToggle={handleToggleAgent}
              onDelete={handleDeleteAgent}
              onUpdate={handleUpdateAgent}
              onRefresh={refresh}
            />
          )}

          {hasAgents && (
            <div className="mt-8">
              <AgentErrorBoundary>
                <AgentDashboard agents={agents} />
              </AgentErrorBoundary>
            </div>
          )}

          <CreateAgentModal
            open={createModal.isOpen}
            onClose={createModal.close}
            onCreate={handleCreateAgent}
          />
        </div>
      </div>
    </AgentErrorBoundary>
  );
};

export default ContinuousAgentPage;
