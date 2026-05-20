'use client';

import { FiChevronLeft, FiChevronRight } from 'react-icons/fi';
import { cn } from '@/utils/classNames';
import { Button } from './Button';

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  showFirstLast?: boolean;
  maxVisible?: number;
  className?: string;
}

export function Pagination({
  currentPage,
  totalPages,
  onPageChange,
  showFirstLast = false,
  maxVisible = 5,
  className,
}: PaginationProps) {
  if (totalPages <= 1) return null;

  const getPageNumbers = () => {
    const pages: (number | string)[] = [];
    const half = Math.floor(maxVisible / 2);

    let start = Math.max(1, currentPage - half);
    let end = Math.min(totalPages, currentPage + half);

    if (end - start < maxVisible - 1) {
      if (start === 1) {
        end = Math.min(totalPages, start + maxVisible - 1);
      } else {
        start = Math.max(1, end - maxVisible + 1);
      }
    }

    if (showFirstLast && start > 1) {
      pages.push(1);
      if (start > 2) pages.push('...');
    }

    for (let i = start; i <= end; i++) {
      pages.push(i);
    }

    if (showFirstLast && end < totalPages) {
      if (end < totalPages - 1) pages.push('...');
      pages.push(totalPages);
    }

    return pages;
  };

  return (
    <nav className={cn('flex items-center justify-center gap-1', className)} aria-label="Pagination">
      <Button
        variant="ghost"
        size="sm"
        onClick={() => onPageChange(currentPage - 1)}
        disabled={currentPage === 1}
        aria-label="Página anterior"
      >
        <FiChevronLeft size={18} />
      </Button>

      {getPageNumbers().map((page, index) => {
        if (page === '...') {
          return (
            <span key={`ellipsis-${index}`} className="px-2 text-gray-500">
              ...
            </span>
          );
        }

        const pageNum = page as number;
        return (
          <Button
            key={pageNum}
            variant={currentPage === pageNum ? 'primary' : 'ghost'}
            size="sm"
            onClick={() => onPageChange(pageNum)}
            className={cn(
              'min-w-[40px]',
              currentPage === pageNum && 'font-semibold'
            )}
            aria-label={`Página ${pageNum}`}
            aria-current={currentPage === pageNum ? 'page' : undefined}
          >
            {pageNum}
          </Button>
        );
      })}

      <Button
        variant="ghost"
        size="sm"
        onClick={() => onPageChange(currentPage + 1)}
        disabled={currentPage === totalPages}
        aria-label="Página siguiente"
      >
        <FiChevronRight size={18} />
      </Button>
    </nav>
  );
}

