"use client";

import React from "react";
import { Input } from "../ui/Input";
import { Select } from "../ui/Select";
import { Button } from "../ui/Button";
import type { FilterStatus, SortField } from "../AgentFilters";

type FilterControlsProps = {
  readonly searchQuery: string;
  readonly onSearchChange: (query: string) => void;
  readonly filterStatus: FilterStatus;
  readonly onFilterChange: (status: FilterStatus) => void;
  readonly sortField: SortField;
  readonly sortOrder: "asc" | "desc";
  readonly onSortFieldChange: (field: SortField) => void;
  readonly onSortOrderToggle: () => void;
  readonly onClearFilters: () => void;
  readonly hasActiveFilters: boolean;
};

/**
 * Filter controls component extracted from AgentFilters
 * for better organization
 */
export const FilterControls = ({
  searchQuery,
  onSearchChange,
  filterStatus,
  onFilterChange,
  sortField,
  sortOrder,
  onSortFieldChange,
  onSortOrderToggle,
  onClearFilters,
  hasActiveFilters,
}: FilterControlsProps): JSX.Element => {
  return (
    <div className="flex flex-wrap gap-4 items-end">
      {/* Search */}
      <div className="flex-1 min-w-[200px]">
        <label htmlFor="agent-search" className="block text-sm font-medium text-gray-700 mb-1">
          Buscar
        </label>
        <Input
          id="agent-search"
          type="text"
          placeholder="Buscar por nombre o descripción..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="w-full"
          ariaLabel="Buscar agentes"
        />
      </div>

      {/* Status Filter */}
      <div className="min-w-[150px]">
        <label htmlFor="filter-status" className="block text-sm font-medium text-gray-700 mb-1">
          Estado
        </label>
        <Select
          id="filter-status"
          value={filterStatus}
          onChange={(e) => onFilterChange(e.target.value as FilterStatus)}
          options={[
            { value: "all", label: "Todos" },
            { value: "active", label: "Activos" },
            { value: "inactive", label: "Inactivos" },
          ]}
          ariaLabel="Filtrar por estado"
        />
      </div>

      {/* Sort Field */}
      <div className="min-w-[180px]">
        <label htmlFor="sort-field" className="block text-sm font-medium text-gray-700 mb-1">
          Ordenar por
        </label>
        <Select
          id="sort-field"
          value={sortField}
          onChange={(e) => onSortFieldChange(e.target.value as SortField)}
          options={[
            { value: "name", label: "Nombre" },
            { value: "createdAt", label: "Fecha creación" },
            { value: "lastExecution", label: "Última ejecución" },
            { value: "status", label: "Estado" },
          ]}
          ariaLabel="Ordenar agentes"
        />
      </div>

      {/* Sort Order Indicator */}
      <div className="flex items-end">
        <button
          type="button"
          onClick={onSortOrderToggle}
          className="px-3 py-2 border border-gray-300 rounded-md bg-white hover:bg-gray-50 transition-colors text-sm font-medium text-gray-700"
          aria-label={`Orden ${sortOrder === "asc" ? "ascendente" : "descendente"}`}
          title={`Cambiar a orden ${sortOrder === "asc" ? "descendente" : "ascendente"}`}
        >
          {sortOrder === "asc" ? "↑" : "↓"}
        </button>
      </div>

      {/* Clear Filters */}
      {hasActiveFilters && (
        <div className="flex items-end">
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={onClearFilters}
            ariaLabel="Limpiar filtros"
          >
            Limpiar
          </Button>
        </div>
      )}
    </div>
  );
};



