import React, { useCallback, useMemo } from "react";
import { cn } from "../utils/classNames";
import { FilterControls } from "./AgentFilters/FilterControls";
import { ActiveFiltersSummary } from "./AgentFilters/ActiveFiltersSummary";

export type FilterStatus = "all" | "active" | "inactive";
export type SortField = "name" | "createdAt" | "lastExecution" | "status";
export type SortOrder = "asc" | "desc";

type AgentFiltersProps = {
  readonly searchQuery: string;
  readonly onSearchChange: (query: string) => void;
  readonly filterStatus: FilterStatus;
  readonly onFilterChange: (status: FilterStatus) => void;
  readonly sortField: SortField;
  readonly sortOrder: SortOrder;
  readonly onSortChange: (field: SortField, order?: SortOrder) => void;
  readonly className?: string;
};

/**
 * Component for filtering and sorting agents
 * 
 * Features:
 * - Search by name or description
 * - Filter by status (all/active/inactive)
 * - Sort by various fields
 * - Clear filters button
 * 
 * @param props - Component props
 * @returns The rendered filters component
 */
export const AgentFilters = ({
  searchQuery,
  onSearchChange,
  filterStatus,
  onFilterChange,
  sortField,
  sortOrder,
  onSortChange,
  className,
}: AgentFiltersProps): JSX.Element => {
  const hasActiveFilters = useMemo(() => {
    return searchQuery.trim() !== "" || filterStatus !== "all";
  }, [searchQuery, filterStatus]);

  const handleClearFilters = useCallback((): void => {
    onSearchChange("");
    onFilterChange("all");
    onSortChange("createdAt", "desc");
  }, [onSearchChange, onFilterChange, onSortChange]);

  const handleSortFieldChange = useCallback(
    (field: SortField): void => {
      // Toggle order if clicking same field, otherwise default to desc
      const newOrder =
        field === sortField && sortOrder === "desc" ? "asc" : "desc";
      onSortChange(field, newOrder);
    },
    [sortField, sortOrder, onSortChange]
  );

  const handleSortOrderToggle = useCallback((): void => {
    onSortChange(sortField, sortOrder === "asc" ? "desc" : "asc");
  }, [sortField, sortOrder, onSortChange]);

  return (
    <div className={cn("space-y-4", className)}>
      <FilterControls
        searchQuery={searchQuery}
        onSearchChange={onSearchChange}
        filterStatus={filterStatus}
        onFilterChange={onFilterChange}
        sortField={sortField}
        sortOrder={sortOrder}
        onSortFieldChange={handleSortFieldChange}
        onSortOrderToggle={handleSortOrderToggle}
        onClearFilters={handleClearFilters}
        hasActiveFilters={hasActiveFilters}
      />

      {hasActiveFilters && (
        <ActiveFiltersSummary
          searchQuery={searchQuery}
          filterStatus={filterStatus}
          sortField={sortField}
          sortOrder={sortOrder}
        />
      )}
    </div>
  );
};


