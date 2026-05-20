"use client";

import Link from "next/link";

interface EmptyStateProps {
  hasFilters: boolean;
  onClearFilters?: () => void;
}

export function EmptyState({ hasFilters, onClearFilters }: EmptyStateProps) {
  return (
    <div className="text-center py-8">
      <p className="text-sm text-gray-500 mb-2">
        {hasFilters
          ? "No se encontraron agentes con los filtros aplicados"
          : "No hay agentes configurados"}
      </p>
      {hasFilters && onClearFilters ? (
        <button
          onClick={onClearFilters}
          className="text-xs text-blue-600 hover:text-blue-700"
        >
          Limpiar filtros
        </button>
      ) : !hasFilters ? (
        <Link
          href="/continuous-agent"
          className="text-xs text-blue-600 hover:text-blue-700 inline-flex items-center gap-1"
        >
          Crear primer agente
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5l7 7-7 7"
            />
          </svg>
        </Link>
      ) : null}
    </div>
  );
}
