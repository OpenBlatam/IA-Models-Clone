/**
 * Agent-specific formatting utilities
 */

import { formatDateTime } from "./dateUtils";

/**
 * Formats next execution date for display
 * @param nextExecutionAt - ISO date string or null
 * @returns Formatted date string or "No programado"
 */
export const formatNextExecutionDate = (nextExecutionAt: string | null): string => {
  if (!nextExecutionAt) {
    return "No programado";
  }
  return formatDateTime(nextExecutionAt);
};

/**
 * Formats agent status text
 * @param isActive - Whether the agent is active
 * @returns Status text with emoji
 */
export const formatAgentStatus = (isActive: boolean): string => {
  return isActive ? "🟢 Activo" : "⚫ Inactivo";
};

/**
 * Gets status class name based on active state
 * @param isActive - Whether the agent is active
 * @returns CSS class name for status
 */
export const getStatusClass = (isActive: boolean): string => {
  return isActive ? "text-green-600" : "text-gray-600";
};



