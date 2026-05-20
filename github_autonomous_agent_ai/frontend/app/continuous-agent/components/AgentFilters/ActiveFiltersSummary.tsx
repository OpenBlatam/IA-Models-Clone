"use client";

import React from "react";
import type { FilterStatus, SortField } from "../AgentFilters";

type ActiveFiltersSummaryProps = {
  readonly searchQuery: string;
  readonly filterStatus: FilterStatus;
  readonly sortField: SortField;
  readonly sortOrder: "asc" | "desc";
};

const SORT_FIELD_LABELS: Record<SortField, string> = {
  name: "Nombre",
  createdAt: "Fecha",
  lastExecution: "Ejecución",
  status: "Estado",
};

/**
 * Active filters summary component
 * Shows current filter and sort state
 */
export const ActiveFiltersSummary = ({
  searchQuery,
  filterStatus,
  sortField,
  sortOrder,
}: ActiveFiltersSummaryProps): JSX.Element => {
  return (
    <div className="flex items-center gap-2 text-sm text-gray-600">
      <span className="font-medium">Filtros activos:</span>
      {searchQuery && (
        <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded">
          Búsqueda: "{searchQuery}"
        </span>
      )}
      {filterStatus !== "all" && (
        <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded">
          Estado: {filterStatus === "active" ? "Activos" : "Inactivos"}
        </span>
      )}
      <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded">
        Orden: {SORT_FIELD_LABELS[sortField]} ({sortOrder === "asc" ? "↑" : "↓"})
      </span>
    </div>
  );
};



