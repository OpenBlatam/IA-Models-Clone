export type ViewMode = 'list' | 'table' | 'calendar';

export type NotificationType = 'success' | 'error' | 'warning' | 'info';

export interface PaginationParams {
  page?: number;
  limit?: number;
  offset?: number;
}

export interface PaginationResponse {
  total_items: number;
  total_pages: number;
  current_page: number;
  items_per_page: number;
}

export interface SearchParams {
  query?: string;
  fields?: string[];
}

export interface DateRange {
  start: Date | null;
  end: Date | null;
}

export interface FilterParams {
  status?: string[];
  dateRange?: DateRange;
  businessArea?: string[];
  tags?: string[];
}

