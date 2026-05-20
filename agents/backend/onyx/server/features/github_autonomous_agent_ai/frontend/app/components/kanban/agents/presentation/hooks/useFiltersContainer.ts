import { useState, useMemo } from "react";
import { filterAgents } from "../../utils/filters";
import { container } from "../../infrastructure/container/DIContainer";
import type { AgentEntity } from "../../domain/entities/Agent";
import type { FilterStatus } from "../../types";

export function useFiltersContainer(agents: AgentEntity[]) {
  const [searchQuery, setSearchQuery] = useState("");
  const [filterStatus, setFilterStatus] = useState<FilterStatus>("all");

  const filteredAgents = useMemo(() => {
    // Ensure agents is always an array
    const agentsArray = Array.isArray(agents) ? agents : [];
    
    // Convertir a DTOs solo para filtrar
    const dtos = container.adapter.toDTOArray(agentsArray);
    const filtered = filterAgents(dtos, searchQuery, filterStatus);
    
    // Mapear de vuelta a entidades manteniendo referencias originales
    const entityMap = new Map(agentsArray.map((e) => [e.id, e]));
    return filtered
      .map((dto) => entityMap.get(dto.id))
      .filter((entity): entity is AgentEntity => entity !== undefined);
  }, [agents, searchQuery, filterStatus]);

  const clearFilters = () => {
    setSearchQuery("");
    setFilterStatus("all");
  };

  const hasActiveFilters = searchQuery.trim() !== "" || filterStatus !== "all";

  return {
    searchQuery,
    setSearchQuery,
    filterStatus,
    setFilterStatus,
    filteredAgents,
    clearFilters,
    hasActiveFilters,
  };
}
