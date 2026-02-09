import { useState, useMemo, useCallback } from "react";
import { useAgentsContext } from "../context/AgentsContext";
import { useFiltersContainer } from "./useFiltersContainer";
import { useAgentStatsContainer } from "./useAgentStatsContainer";
import { useAgentActionsContainer } from "./useAgentActionsContainer";
import { selectAgentDTOs, createEntityMap } from "../selectors/agentSelectors";
import { createToggleHandler } from "../factories/handlerFactory";
import type { ViewMode } from "../../../types";
import type { AgentEntity } from "../../../domain/entities/Agent";

/**
 * Hook compuesto que encapsula toda la lógica del AgentsSection
 * Simplifica el uso y reduce la complejidad del componente
 */
export function useAgentsSection() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("cards");

  const { agents: agentEntities, isLoading, refetch } = useAgentsContext();
  const {
    searchQuery,
    setSearchQuery,
    filterStatus,
    setFilterStatus,
    filteredAgents: filteredEntities,
    clearFilters,
    hasActiveFilters,
  } = useFiltersContainer(agentEntities);

  const stats = useAgentStatsContainer(agentEntities);
  const { toggleActive } = useAgentActionsContainer(refetch);

  // Memoizar conversión a DTOs usando selectors
  const agents = useMemo(() => selectAgentDTOs(agentEntities), [agentEntities]);
  const filteredAgents = useMemo(() => selectAgentDTOs(filteredEntities), [filteredEntities]);

  // Crear mapa de entidades una sola vez
  const entityMap = useMemo(() => createEntityMap(agentEntities), [agentEntities]);

  // Handler optimizado usando factory
  const handleToggleActive = useMemo(
    () => createToggleHandler(entityMap, toggleActive),
    [entityMap, toggleActive]
  );

  const toggleExpand = useCallback(() => setIsExpanded((prev) => !prev), []);
  const toggleViewMode = useCallback(() => setViewMode((prev) => (prev === "table" ? "cards" : "table")), []);

  return {
    // State
    isExpanded,
    viewMode,
    isLoading,
    
    // Data
    agents,
    filteredAgents,
    stats,
    agentCount: agents.length,
    
    // Filters
    searchQuery,
    setSearchQuery,
    filterStatus,
    setFilterStatus,
    hasActiveFilters,
    clearFilters,
    
    // Actions
    toggleExpand,
    toggleViewMode,
    handleToggleActive,
  };
}
