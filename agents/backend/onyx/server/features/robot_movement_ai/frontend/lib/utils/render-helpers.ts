/**
 * Render helper utilities
 */

import { ReactNode } from 'react';

/**
 * Conditional render
 */
export function renderIf(
  condition: boolean,
  render: () => ReactNode
): ReactNode {
  return condition ? render() : null;
}

/**
 * Render with fallback
 */
export function renderWithFallback(
  condition: boolean,
  render: () => ReactNode,
  fallback: () => ReactNode
): ReactNode {
  return condition ? render() : fallback();
}

/**
 * Render list with empty state
 */
export function renderList<T>(
  items: T[],
  renderItem: (item: T, index: number) => ReactNode,
  renderEmpty?: () => ReactNode
): ReactNode {
  if (items.length === 0) {
    return renderEmpty ? renderEmpty() : null;
  }
  return items.map(renderItem);
}

/**
 * Render loading state
 */
export function renderLoading(
  isLoading: boolean,
  renderContent: () => ReactNode,
  renderLoader?: () => ReactNode
): ReactNode {
  if (isLoading) {
    return renderLoader ? renderLoader() : <div>Loading...</div>;
  }
  return renderContent();
}

/**
 * Render error state
 */
export function renderError(
  error: Error | null,
  renderContent: () => ReactNode,
  renderError?: (error: Error) => ReactNode
): ReactNode {
  if (error) {
    return renderError
      ? renderError(error)
      : <div>Error: {error.message}</div>;
  }
  return renderContent();
}

/**
 * Render with multiple conditions
 */
export function renderWithConditions(
  conditions: Array<{ condition: boolean; render: () => ReactNode }>,
  fallback?: () => ReactNode
): ReactNode {
  for (const { condition, render } of conditions) {
    if (condition) {
      return render();
    }
  }
  return fallback ? fallback() : null;
}



