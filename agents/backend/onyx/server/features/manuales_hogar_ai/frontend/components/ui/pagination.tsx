'use client';

import { Button } from './button';
import type { PaginationProps } from '@/lib/types/components';

export const Pagination = ({
  currentPage,
  totalItems,
  itemsPerPage,
  onPrevious,
  onNext,
  hasMore,
}: PaginationProps): JSX.Element => {
  const canGoPrevious = currentPage > 1;
  const canGoNext = hasMore !== undefined ? hasMore : totalItems >= itemsPerPage;

  return (
    <div className="flex items-center justify-between pt-4">
      <Button
        variant="outline"
        onClick={onPrevious}
        disabled={!canGoPrevious}
        aria-label="Página anterior"
      >
        Anterior
      </Button>
      <span className="text-sm text-gray-600">
        Página {currentPage}
      </span>
      <Button
        variant="outline"
        onClick={onNext}
        disabled={!canGoNext}
        aria-label="Página siguiente"
      >
        Siguiente
      </Button>
    </div>
  );
};

