/**
 * Utilities for filtering and sorting agents
 */

import type { ContinuousAgent } from "../types";
import type { FilterStatus, SortField, SortOrder } from "../components/AgentFilters";

/**
 * Filters agents based on search query and status
 */
export const filterAgents = (
  agents: ContinuousAgent[],
  searchQuery: string,
  filterStatus: FilterStatus
): ContinuousAgent[] => {
  return agents.filter((agent) => {
    // Status filter
    if (filterStatus === "active" && !agent.isActive) return false;
    if (filterStatus === "inactive" && agent.isActive) return false;

    // Search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      const matchesName = agent.name.toLowerCase().includes(query);
      const matchesDescription = agent.description.toLowerCase().includes(query);
      const matchesTaskType = agent.config.taskType.toLowerCase().includes(query);
      
      // Also search in goal if present
      const matchesGoal = agent.config.goal?.toLowerCase().includes(query) || false;

      if (!matchesName && !matchesDescription && !matchesTaskType && !matchesGoal) {
        return false;
      }
    }

    return true;
  });
};

/**
 * Sorts agents by the specified field and order
 */
export const sortAgents = (
  agents: ContinuousAgent[],
  sortField: SortField,
  sortOrder: SortOrder
): ContinuousAgent[] => {
  return [...agents].sort((a, b) => {
    let comparison = 0;

    switch (sortField) {
      case "name":
        comparison = a.name.localeCompare(b.name);
        break;
      case "createdAt":
        comparison =
          new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
        break;
      case "lastExecution":
        const aTime = a.stats.lastExecutionAt
          ? new Date(a.stats.lastExecutionAt).getTime()
          : 0;
        const bTime = b.stats.lastExecutionAt
          ? new Date(b.stats.lastExecutionAt).getTime()
          : 0;
        comparison = aTime - bTime;
        break;
      case "status":
        comparison = Number(a.isActive) - Number(b.isActive);
        break;
      default:
        return 0;
    }

    return sortOrder === "asc" ? comparison : -comparison;
  });
};

/**
 * Filters and sorts agents in one operation
 */
export const filterAndSortAgents = (
  agents: ContinuousAgent[],
  searchQuery: string,
  filterStatus: FilterStatus,
  sortField: SortField,
  sortOrder: SortOrder
): ContinuousAgent[] => {
  const filtered = filterAgents(agents, searchQuery, filterStatus);
  return sortAgents(filtered, sortField, sortOrder);
};




