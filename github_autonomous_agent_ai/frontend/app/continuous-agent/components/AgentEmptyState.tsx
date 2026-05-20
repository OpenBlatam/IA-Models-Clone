"use client";

import React from "react";
import { STATE_STYLES } from "../constants/state-styles";
import { UI_MESSAGES } from "../constants/messages";

type AgentEmptyStateProps = {
  readonly hasFilters: boolean;
  readonly onClearFilters?: () => void;
  readonly onCreateAgent?: () => void;
};

/**
 * Empty state component when no agents are found
 */
export const AgentEmptyState = ({
  hasFilters,
  onClearFilters,
  onCreateAgent,
}: AgentEmptyStateProps): JSX.Element => {
  return (
    <div className={STATE_STYLES.CENTERED}>
      <p className={STATE_STYLES.EMPTY_MESSAGE}>
        {hasFilters
          ? "No se encontraron agentes que coincidan con los filtros"
          : UI_MESSAGES.NO_AGENTS}
      </p>
      {hasFilters && onClearFilters ? (
        <button
          type="button"
          onClick={onClearFilters}
          className={STATE_STYLES.BUTTON_SECONDARY}
        >
          Limpiar filtros
        </button>
      ) : (
        onCreateAgent && (
          <button
            type="button"
            onClick={onCreateAgent}
            className={STATE_STYLES.BUTTON_PRIMARY}
            aria-label={UI_MESSAGES.CREATE_FIRST_AGENT}
            tabIndex={0}
          >
            {UI_MESSAGES.CREATE_FIRST_AGENT}
          </button>
        )
      )}
    </div>
  );
};

