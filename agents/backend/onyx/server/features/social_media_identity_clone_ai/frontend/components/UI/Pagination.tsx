import { cn } from '@/lib/utils';
import Button from './Button';

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  className?: string;
}

const Pagination = ({
  currentPage,
  totalPages,
  onPageChange,
  className = '',
}: PaginationProps): JSX.Element => {
  const handlePrevious = (): void => {
    if (currentPage > 1) {
      onPageChange(currentPage - 1);
    }
  };

  const handleNext = (): void => {
    if (currentPage < totalPages) {
      onPageChange(currentPage + 1);
    }
  };

  const handlePageClick = (page: number): void => {
    if (page !== currentPage && page >= 1 && page <= totalPages) {
      onPageChange(page);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>, page: number): void => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handlePageClick(page);
    }
  };

  if (totalPages <= 1) {
    return <></>;
  }

  const getPageNumbers = (): number[] => {
    const pages: number[] = [];
    const maxVisible = 5;
    
    if (totalPages <= maxVisible) {
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
      return pages;
    }

    if (currentPage <= 3) {
      for (let i = 1; i <= 4; i++) {
        pages.push(i);
      }
      pages.push(totalPages);
      return pages;
    }

    if (currentPage >= totalPages - 2) {
      pages.push(1);
      for (let i = totalPages - 3; i <= totalPages; i++) {
        pages.push(i);
      }
      return pages;
    }

    pages.push(1);
    for (let i = currentPage - 1; i <= currentPage + 1; i++) {
      pages.push(i);
    }
    pages.push(totalPages);
    return pages;
  };

  const pageNumbers = getPageNumbers();

  return (
    <nav
      className={cn('flex items-center justify-center gap-2', className)}
      role="navigation"
      aria-label="Pagination"
    >
      <Button
        variant="secondary"
        onClick={handlePrevious}
        disabled={currentPage === 1}
        aria-label="Previous page"
        aria-disabled={currentPage === 1}
      >
        Previous
      </Button>

      <div className="flex gap-1">
        {pageNumbers.map((page, index) => {
          const isCurrentPage = page === currentPage;
          const showEllipsis = index > 0 && pageNumbers[index - 1] !== page - 1;

          return (
            <div key={page} className="flex items-center gap-1">
              {showEllipsis && (
                <span className="px-2 text-gray-500" aria-hidden="true">
                  ...
                </span>
              )}
              <button
                onClick={() => handlePageClick(page)}
                onKeyDown={(e) => handleKeyDown(e, page)}
                className={cn(
                  'px-3 py-1 rounded text-sm font-medium transition-colors',
                  isCurrentPage
                    ? 'bg-primary-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-100 border border-gray-300'
                )}
                aria-label={`Page ${page}`}
                aria-current={isCurrentPage ? 'page' : undefined}
                tabIndex={0}
              >
                {page}
              </button>
            </div>
          );
        })}
      </div>

      <Button
        variant="secondary"
        onClick={handleNext}
        disabled={currentPage === totalPages}
        aria-label="Next page"
        aria-disabled={currentPage === totalPages}
      >
        Next
      </Button>
    </nav>
  );
};

export default Pagination;



