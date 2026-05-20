"use client";

import { SearchInput, Select } from "./ui";
import { FILTER_OPTIONS } from "../config/constants";
import type { FilterStatus } from "../types";

interface AgentFiltersProps {
  searchQuery: string;
  filterStatus: FilterStatus;
  onSearchChange: (query: string) => void;
  onFilterStatusChange: (status: FilterStatus) => void;
  onClearFilters: () => void;
}

const STATUS_OPTIONS = [
  { value: FILTER_OPTIONS.STATUS.ALL, label: "Todos los estados" },
  { value: FILTER_OPTIONS.STATUS.ACTIVE, label: "Activos" },
  { value: FILTER_OPTIONS.STATUS.INACTIVE, label: "Inactivos" },
];

export function AgentFilters({
  searchQuery,
  filterStatus,
  onSearchChange,
  onFilterStatusChange,
  onClearFilters,
}: AgentFiltersProps) {
  return (
    <div className="p-4 border-b border-gray-100 bg-gray-50/60 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
      <div className="flex items-center gap-2">
        <SearchInput
          value={searchQuery}
          onChange={onSearchChange}
          placeholder="Buscar por nombre, descripción o tipo..."
        />
        <Select
          value={filterStatus}
          onChange={(value) => onFilterStatusChange(value as FilterStatus)}
          options={STATUS_OPTIONS}
        />
      </div>
      <button
        onClick={onClearFilters}
        className="self-start md:self-auto inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-gray-200 text-xs text-gray-700 hover:bg-gray-100"
      >
        Limpiar filtros
      </button>
    </div>
  );
}
