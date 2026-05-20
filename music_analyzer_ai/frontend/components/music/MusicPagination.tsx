'use client';

import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react';

interface MusicPaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  itemsPerPage?: number;
  onItemsPerPageChange?: (items: number) => void;
}

export function MusicPagination({
  currentPage,
  totalPages,
  onPageChange,
  itemsPerPage,
  onItemsPerPageChange,
}: MusicPaginationProps) {
  const itemsPerPageOptions = [10, 20, 50, 100];

  const getPageNumbers = () => {
    const pages = [];
    const maxVisible = 5;

    if (totalPages <= maxVisible) {
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      if (currentPage <= 3) {
        for (let i = 1; i <= 5; i++) {
          pages.push(i);
        }
        pages.push('...');
        pages.push(totalPages);
      } else if (currentPage >= totalPages - 2) {
        pages.push(1);
        pages.push('...');
        for (let i = totalPages - 4; i <= totalPages; i++) {
          pages.push(i);
        }
      } else {
        pages.push(1);
        pages.push('...');
        for (let i = currentPage - 1; i <= currentPage + 1; i++) {
          pages.push(i);
        }
        pages.push('...');
        pages.push(totalPages);
      }
    }

    return pages;
  };

  return (
    <div className="flex items-center justify-between bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
      <div className="flex items-center gap-2">
        <button
          onClick={() => onPageChange(1)}
          disabled={currentPage === 1}
          className="p-2 text-gray-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
        >
          <ChevronsLeft className="w-4 h-4" />
        </button>
        <button
          onClick={() => onPageChange(currentPage - 1)}
          disabled={currentPage === 1}
          className="p-2 text-gray-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
        >
          <ChevronLeft className="w-4 h-4" />
        </button>
      </div>

      <div className="flex items-center gap-1">
        {getPageNumbers().map((page, idx) => (
          <button
            key={idx}
            onClick={() => typeof page === 'number' && onPageChange(page)}
            disabled={page === '...'}
            className={`px-3 py-1 rounded transition-colors ${
              page === currentPage
                ? 'bg-purple-600 text-white'
                : page === '...'
                ? 'text-gray-400 cursor-default'
                : 'text-gray-300 hover:bg-white/10'
            }`}
          >
            {page}
          </button>
        ))}
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={() => onPageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
          className="p-2 text-gray-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
        >
          <ChevronRight className="w-4 h-4" />
        </button>
        <button
          onClick={() => onPageChange(totalPages)}
          disabled={currentPage === totalPages}
          className="p-2 text-gray-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
        >
          <ChevronsRight className="w-4 h-4" />
        </button>
      </div>

      {itemsPerPage && onItemsPerPageChange && (
        <div className="flex items-center gap-2 ml-4">
          <span className="text-sm text-gray-400">Por página:</span>
          <select
            value={itemsPerPage}
            onChange={(e) => onItemsPerPageChange(parseInt(e.target.value))}
            className="px-2 py-1 bg-white/20 border border-white/30 rounded text-white text-sm focus:outline-none focus:ring-2 focus:ring-purple-400"
          >
            {itemsPerPageOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}


