"use client";

import { useState, useCallback, useMemo } from "react";
import type { FilterStatus, SortField, SortOrder } from "../components/AgentFilters";
import { filterAndSortAgents } from "../utils/agent-filters";
import type { ContinuousAgent } from "../types";

type UseAgentFiltersOptions = {
  readonly agents: readonly ContinuousAgent[];
};

type UseAgentFiltersReturn = {
  readonly searchQuery: string;
  readonly filterStatus: FilterStatus;
  readonly sortField: SortField;
  readonly sortOrder: SortOrder;
  readonly setSearchQuery: (query: string) => void;
  readonly setFilterStatus: (status: FilterStatus) => void;
  readonly setSortField: (field: SortField) => void;
  readonly setSortOrder: (order: SortOrder) => void;
  readonly handleSortChange: (field: SortField, order?: SortOrder) => void;
  readonly clearFilters: () => void;
  readonly filteredAndSortedAgents: readonly ContinuousAgent[];
  readonly hasActiveFilters: boolean;
};

/**
 * Custom hook for managing agent filters and sorting
 * 
 * Features:
 * - Search query filtering
 * - Status filtering (all, active, inactive)
 * - Sorting by field and order
 * - Clear filters functionality
 * - Memoized filtered and sorted results
 */
export const useAgentFilters = ({
  agents,
}: UseAgentFiltersOptions): UseAgentFiltersReturn => {
  const [searchQuery, setSearchQuery] = useState("");
  const [filterStatus, setFilterStatus] = useState<FilterStatus>("all");
  const [sortField, setSortField] = useState<SortField>("createdAt");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");

  const handleSortChange = useCallback(
    (field: SortField, order?: SortOrder): void => {
      setSortField(field);
      if (order !== undefined) {
        setSortOrder(order);
      }
    },
    []
  );

  const clearFilters = useCallback((): void => {
    setSearchQuery("");
    setFilterStatus("all");
  }, []);

  const filteredAndSortedAgents = useMemo(
    () => filterAndSortAgents(agents, searchQuery, filterStatus, sortField, sortOrder),
    [agents, searchQuery, filterStatus, sortField, sortOrder]
  );

  const hasActiveFilters = useMemo(
    () => !!(searchQuery || filterStatus !== "all"),
    [searchQuery, filterStatus]
  );

  return {
    searchQuery,
    filterStatus,
    sortField,
    sortOrder,
    setSearchQuery,
    setFilterStatus,
    setSortField,
    setSortOrder,
    handleSortChange,
    clearFilters,
    filteredAndSortedAgents,
    hasActiveFilters,
  };
};



