/**
 * Pagination utility functions.
 * Provides helper functions for pagination calculations.
 */

/**
 * Pagination options.
 */
export interface PaginationOptions {
  page: number;
  pageSize: number;
  total: number;
}

/**
 * Pagination result.
 */
export interface PaginationResult {
  currentPage: number;
  totalPages: number;
  pageSize: number;
  total: number;
  startIndex: number;
  endIndex: number;
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  nextPage: number | null;
  previousPage: number | null;
}

/**
 * Calculates pagination information.
 * @param options - Pagination options
 * @returns Pagination result
 */
export function calculatePagination(
  options: PaginationOptions
): PaginationResult {
  const { page, pageSize, total } = options;

  const totalPages = Math.ceil(total / pageSize);
  const currentPage = Math.max(1, Math.min(page, totalPages));
  const startIndex = (currentPage - 1) * pageSize;
  const endIndex = Math.min(startIndex + pageSize, total);

  return {
    currentPage,
    totalPages,
    pageSize,
    total,
    startIndex,
    endIndex,
    hasNextPage: currentPage < totalPages,
    hasPreviousPage: currentPage > 1,
    nextPage: currentPage < totalPages ? currentPage + 1 : null,
    previousPage: currentPage > 1 ? currentPage - 1 : null,
  };
}

/**
 * Gets items for current page.
 * @param items - All items
 * @param page - Current page
 * @param pageSize - Items per page
 * @returns Items for current page
 */
export function getPageItems<T>(
  items: T[],
  page: number,
  pageSize: number
): T[] {
  const startIndex = (page - 1) * pageSize;
  const endIndex = startIndex + pageSize;
  return items.slice(startIndex, endIndex);
}

/**
 * Generates page numbers for pagination UI.
 * @param currentPage - Current page
 * @param totalPages - Total pages
 * @param maxVisible - Maximum visible pages (default: 5)
 * @returns Array of page numbers
 */
export function generatePageNumbers(
  currentPage: number,
  totalPages: number,
  maxVisible: number = 5
): (number | 'ellipsis')[] {
  if (totalPages <= maxVisible) {
    return Array.from({ length: totalPages }, (_, i) => i + 1);
  }

  const pages: (number | 'ellipsis')[] = [];
  const half = Math.floor(maxVisible / 2);

  let start = Math.max(1, currentPage - half);
  let end = Math.min(totalPages, start + maxVisible - 1);

  if (end - start < maxVisible - 1) {
    start = Math.max(1, end - maxVisible + 1);
  }

  if (start > 1) {
    pages.push(1);
    if (start > 2) {
      pages.push('ellipsis');
    }
  }

  for (let i = start; i <= end; i++) {
    pages.push(i);
  }

  if (end < totalPages) {
    if (end < totalPages - 1) {
      pages.push('ellipsis');
    }
    pages.push(totalPages);
  }

  return pages;
}

