'use client';

import { memo, useCallback, useMemo } from 'react';
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react';
import { Button } from './Button';
import { cn } from '@/lib/utils';

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  className?: string;
  showFirstLast?: boolean;
  showInfo?: boolean;
  maxVisible?: number;
  siblingCount?: number;
}

const Pagination = memo(
  ({
    currentPage,
    totalPages,
    onPageChange,
    className,
    showFirstLast = false,
    showInfo = false,
    maxVisible = 5,
    siblingCount = 1,
  }: PaginationProps): JSX.Element => {
    const handlePrevious = useCallback((): void => {
      if (currentPage > 1) {
        onPageChange(currentPage - 1);
      }
    }, [currentPage, onPageChange]);

    const handleNext = useCallback((): void => {
      if (currentPage < totalPages) {
        onPageChange(currentPage + 1);
      }
    }, [currentPage, totalPages, onPageChange]);

    const handleFirst = useCallback((): void => {
      onPageChange(1);
    }, [onPageChange]);

    const handleLast = useCallback((): void => {
      onPageChange(totalPages);
    }, [totalPages, onPageChange]);

    const handlePageClick = useCallback(
      (page: number): void => {
        onPageChange(page);
      },
      [onPageChange]
    );

    const getPageNumbers = useMemo((): (number | string)[] => {
      const pages: (number | string)[] = [];
      const totalNumbers = maxVisible;
      const totalBlocks = totalNumbers + 2;

      if (totalPages <= totalBlocks) {
        for (let i = 1; i <= totalPages; i++) {
          pages.push(i);
        }
        return pages;
      }

      const leftSiblingIndex = Math.max(currentPage - siblingCount, 1);
      const rightSiblingIndex = Math.min(currentPage + siblingCount, totalPages);

      const shouldShowLeftDots = leftSiblingIndex > 2;
      const shouldShowRightDots = rightSiblingIndex < totalPages - 1;

      if (!shouldShowLeftDots && shouldShowRightDots) {
        const leftItemCount = 3 + 2 * siblingCount;
        for (let i = 1; i <= leftItemCount; i++) {
          pages.push(i);
        }
        pages.push('...');
        pages.push(totalPages);
      } else if (shouldShowLeftDots && !shouldShowRightDots) {
        pages.push(1);
        pages.push('...');
        const rightItemCount = 3 + 2 * siblingCount;
        for (let i = totalPages - rightItemCount + 1; i <= totalPages; i++) {
          pages.push(i);
        }
      } else {
        pages.push(1);
        pages.push('...');
        for (let i = leftSiblingIndex; i <= rightSiblingIndex; i++) {
          pages.push(i);
        }
        pages.push('...');
        pages.push(totalPages);
      }

      return pages;
    }, [currentPage, totalPages, maxVisible, siblingCount]);

    const pageNumbers = getPageNumbers;

    if (totalPages <= 1) {
      return null;
    }

    const startItem = (currentPage - 1) * 10 + 1;
    const endItem = Math.min(currentPage * 10, totalPages * 10);

    return (
      <nav className={cn('flex items-center justify-center space-x-2', className)} aria-label="Pagination">
        {showFirstLast && (
          <Button
            variant="secondary"
            size="sm"
            onClick={handleFirst}
            disabled={currentPage === 1}
            aria-label="First page"
            tabIndex={0}
          >
            <ChevronsLeft className="w-4 h-4" aria-hidden="true" />
          </Button>
        )}

        <Button
          variant="secondary"
          size="sm"
          onClick={handlePrevious}
          disabled={currentPage === 1}
          aria-label="Previous page"
          tabIndex={0}
        >
          <ChevronLeft className="w-4 h-4" aria-hidden="true" />
        </Button>

        {pageNumbers.map((page, index) => {
          if (page === '...') {
            return (
              <span
                key={`ellipsis-${index}`}
                className="px-2 text-gray-400"
                aria-hidden="true"
              >
                ...
              </span>
            );
          }

          const pageNum = page as number;
          return (
            <Button
              key={pageNum}
              variant={pageNum === currentPage ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => handlePageClick(pageNum)}
              aria-label={`Page ${pageNum}`}
              aria-current={pageNum === currentPage ? 'page' : undefined}
              tabIndex={0}
            >
              {pageNum}
            </Button>
          );
        })}

        <Button
          variant="secondary"
          size="sm"
          onClick={handleNext}
          disabled={currentPage === totalPages}
          aria-label="Next page"
          tabIndex={0}
        >
          <ChevronRight className="w-4 h-4" aria-hidden="true" />
        </Button>

        {showFirstLast && (
          <Button
            variant="secondary"
            size="sm"
            onClick={handleLast}
            disabled={currentPage === totalPages}
            aria-label="Last page"
            tabIndex={0}
          >
            <ChevronsRight className="w-4 h-4" aria-hidden="true" />
          </Button>
        )}

        {showInfo && (
          <span className="ml-4 text-sm text-gray-600" aria-live="polite">
            Page {currentPage} of {totalPages}
          </span>
        )}
      </nav>
    );
  }
);

Pagination.displayName = 'Pagination';

export default Pagination;

