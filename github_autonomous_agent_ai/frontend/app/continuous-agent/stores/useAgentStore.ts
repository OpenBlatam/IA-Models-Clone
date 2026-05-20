/**
 * Zustand store for global agent state management
 * 
 * Provides centralized state for:
 * - UI state (modals, filters, sorting)
 * - Selected agents
 * - View preferences
 * - Cached data
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { ContinuousAgent } from "../types";

export type ViewMode = "grid" | "list";
export type SortField = "name" | "createdAt" | "lastExecution" | "status";
export type SortOrder = "asc" | "desc";
export type FilterStatus = "all" | "active" | "inactive";

interface AgentUIState {
  // View preferences
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;

  // Sorting
  sortField: SortField;
  sortOrder: SortOrder;
  setSorting: (field: SortField, order?: SortOrder) => void;

  // Filtering
  filterStatus: FilterStatus;
  setFilterStatus: (status: FilterStatus) => void;
  searchQuery: string;
  setSearchQuery: (query: string) => void;

  // Selected agents
  selectedAgentIds: Set<string>;
  toggleAgentSelection: (agentId: string) => void;
  selectAllAgents: (agentIds: string[]) => void;
  clearSelection: () => void;

  // UI state
  isCreateModalOpen: boolean;
  setIsCreateModalOpen: (open: boolean) => void;
  isDeleteModalOpen: boolean;
  setIsDeleteModalOpen: (open: boolean) => void;
  isSettingsOpen: boolean;
  setIsSettingsOpen: (open: boolean) => void;

  // Cached data
  lastFetchedAt: number | null;
  setLastFetchedAt: (timestamp: number) => void;

  // Bulk actions
  isBulkActionMode: boolean;
  setIsBulkActionMode: (enabled: boolean) => void;
}

/**
 * Zustand store for agent UI state
 * 
 * Features:
 * - Persistent view preferences (localStorage)
 * - Centralized UI state management
 * - Optimized selectors for performance
 * - Type-safe state updates
 */
export const useAgentStore = create<AgentUIState>()(
  persist(
    (set) => ({
      // View preferences
      viewMode: "grid",
      setViewMode: (mode) => set({ viewMode: mode }),

      // Sorting
      sortField: "createdAt",
      sortOrder: "desc",
      setSorting: (field, order) =>
        set((state) => ({
          sortField: field,
          sortOrder: order ?? (state.sortField === field && state.sortOrder === "asc" ? "desc" : "asc"),
        })),

      // Filtering
      filterStatus: "all",
      setFilterStatus: (status) => set({ filterStatus: status }),
      searchQuery: "",
      setSearchQuery: (query) => set({ searchQuery: query }),

      // Selected agents
      selectedAgentIds: new Set<string>(),
      toggleAgentSelection: (agentId) =>
        set((state) => {
          const newSet = new Set(state.selectedAgentIds);
          if (newSet.has(agentId)) {
            newSet.delete(agentId);
          } else {
            newSet.add(agentId);
          }
          return { selectedAgentIds: newSet };
        }),
      selectAllAgents: (agentIds) =>
        set({ selectedAgentIds: new Set(agentIds) }),
      clearSelection: () => set({ selectedAgentIds: new Set() }),

      // UI state
      isCreateModalOpen: false,
      setIsCreateModalOpen: (open) => set({ isCreateModalOpen: open }),
      isDeleteModalOpen: false,
      setIsDeleteModalOpen: (open) => set({ isDeleteModalOpen: open }),
      isSettingsOpen: false,
      setIsSettingsOpen: (open) => set({ isSettingsOpen: open }),

      // Cached data
      lastFetchedAt: null,
      setLastFetchedAt: (timestamp) => set({ lastFetchedAt: timestamp }),

      // Bulk actions
      isBulkActionMode: false,
      setIsBulkActionMode: (enabled) => set({ isBulkActionMode: enabled }),
    }),
    {
      name: "continuous-agent-ui-storage",
      partialize: (state) => ({
        // Only persist view preferences, not UI state
        viewMode: state.viewMode,
        sortField: state.sortField,
        sortOrder: state.sortOrder,
        filterStatus: state.filterStatus,
      }),
    }
  )
);

/**
 * Optimized selectors for better performance
 */
export const useViewMode = () => useAgentStore((state) => state.viewMode);
export const useSorting = () =>
  useAgentStore((state) => ({
    field: state.sortField,
    order: state.sortOrder,
  }));
export const useFilters = () =>
  useAgentStore((state) => ({
    status: state.filterStatus,
    searchQuery: state.searchQuery,
  }));
export const useSelectedAgents = () =>
  useAgentStore((state) => state.selectedAgentIds);
export const useIsBulkActionMode = () =>
  useAgentStore((state) => state.isBulkActionMode);

/**
 * Helper function to filter and sort agents
 */
export const useFilteredAndSortedAgents = (
  agents: ContinuousAgent[]
): ContinuousAgent[] => {
  const { status, searchQuery } = useFilters();
  const { field, order } = useSorting();

  return agents
    .filter((agent) => {
      // Status filter
      if (status === "active" && !agent.isActive) return false;
      if (status === "inactive" && agent.isActive) return false;

      // Search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        return (
          agent.name.toLowerCase().includes(query) ||
          agent.description.toLowerCase().includes(query)
        );
      }

      return true;
    })
    .sort((a, b) => {
      let comparison = 0;

      switch (field) {
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

      return order === "asc" ? comparison : -comparison;
    });
};

