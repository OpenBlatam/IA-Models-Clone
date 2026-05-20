/**
 * Type Tests - Common Types
 * Tests to ensure type definitions are correct
 */

import type {
  ApiResponse,
  PaginatedResponse,
  PaginationMeta,
  SortConfig,
  FilterConfig,
  ViewMode,
  LoadingState,
  AsyncState,
} from '@/lib/types/common';

// These are type tests, so we're checking that types compile correctly
// and that the structure matches expectations

describe('Common Types', () => {
  describe('ApiResponse', () => {
    it('should have correct structure for success response', () => {
      const successResponse: ApiResponse<{ data: string }> = {
        success: true,
        data: { data: 'test' },
      };

      expect(successResponse.success).toBe(true);
      expect(successResponse.data).toBeDefined();
    });

    it('should have correct structure for error response', () => {
      const errorResponse: ApiResponse<never> = {
        success: false,
        error: 'Error message',
      };

      expect(errorResponse.success).toBe(false);
      expect(errorResponse.error).toBeDefined();
    });

    it('should support message field', () => {
      const response: ApiResponse<string> = {
        success: true,
        data: 'test',
        message: 'Success message',
      };

      expect(response.message).toBeDefined();
    });
  });

  describe('PaginationMeta', () => {
    it('should have all required fields', () => {
      const meta: PaginationMeta = {
        page: 1,
        limit: 20,
        total: 100,
        totalPages: 5,
        hasNext: true,
        hasPrev: false,
      };

      expect(meta.page).toBe(1);
      expect(meta.limit).toBe(20);
      expect(meta.total).toBe(100);
      expect(meta.totalPages).toBe(5);
      expect(meta.hasNext).toBe(true);
      expect(meta.hasPrev).toBe(false);
    });
  });

  describe('PaginatedResponse', () => {
    it('should have pagination metadata', () => {
      const paginatedResponse: PaginatedResponse<string> = {
        data: ['item1', 'item2'],
        meta: {
          page: 1,
          limit: 20,
          total: 100,
          totalPages: 5,
          hasNext: true,
          hasPrev: false,
        },
      };

      expect(paginatedResponse.data).toBeDefined();
      expect(paginatedResponse.meta).toBeDefined();
      expect(paginatedResponse.meta.page).toBe(1);
      expect(paginatedResponse.meta.totalPages).toBe(5);
    });
  });

  describe('SortConfig', () => {
    it('should have field and order', () => {
      const sortConfig: SortConfig = {
        field: 'name',
        order: 'asc',
      };

      expect(sortConfig.field).toBe('name');
      expect(sortConfig.order).toBe('asc');
    });

    it('should support desc order', () => {
      const sortConfig: SortConfig = {
        field: 'popularity',
        order: 'desc',
      };

      expect(sortConfig.order).toBe('desc');
    });
  });

  describe('FilterConfig', () => {
    it('should accept any key-value pairs', () => {
      const filterConfig: FilterConfig = {
        genre: 'rock',
        year: 2020,
        popularity: 80,
      };

      expect(filterConfig.genre).toBe('rock');
      expect(filterConfig.year).toBe(2020);
    });
  });

  describe('ViewMode', () => {
    it('should support all view modes', () => {
      const gridMode: ViewMode = 'grid';
      const listMode: ViewMode = 'list';
      const compactMode: ViewMode = 'compact';

      expect(gridMode).toBe('grid');
      expect(listMode).toBe('list');
      expect(compactMode).toBe('compact');
    });
  });

  describe('LoadingState', () => {
    it('should have loading and error fields', () => {
      const loadingState: LoadingState = {
        isLoading: true,
        error: null,
      };

      expect(loadingState.isLoading).toBe(true);
      expect(loadingState.error).toBeNull();
    });

    it('should support error state', () => {
      const error = new Error('Test error');
      const loadingState: LoadingState = {
        isLoading: false,
        error,
      };

      expect(loadingState.error).toBe(error);
    });
  });

  describe('AsyncState', () => {
    it('should extend LoadingState and add data', () => {
      const asyncState: AsyncState<string> = {
        isLoading: false,
        error: null,
        data: 'test data',
      };

      expect(asyncState.data).toBe('test data');
      expect(asyncState.isLoading).toBe(false);
    });

    it('should support null data', () => {
      const asyncState: AsyncState<string> = {
        isLoading: true,
        error: null,
        data: null,
      };

      expect(asyncState.data).toBeNull();
    });
  });
});

