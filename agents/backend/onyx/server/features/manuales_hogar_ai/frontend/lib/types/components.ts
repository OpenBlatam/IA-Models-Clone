import type { Category, ManualListItem, ManualDetailResponse, Rating } from './api';

export interface ManualListProps {
  manuals?: ManualListItem[];
  isLoading?: boolean;
  error?: Error | null;
  emptyMessage?: string;
}

export interface CategorySelectProps {
  value?: Category | string;
  onValueChange: (value: Category | string | undefined) => void;
  placeholder?: string;
  includeAll?: boolean;
}

export interface ModelSelectProps {
  value?: string;
  onValueChange: (value: string | undefined) => void;
  placeholder?: string;
}

export interface FileUploadProps {
  files: File[];
  onFilesChange: (files: File[]) => void;
  maxFiles?: number;
  accept?: string;
  label?: string;
  multiple?: boolean;
}

export interface StarRatingProps {
  rating: number;
  onRatingChange?: (rating: number) => void;
  readonly?: boolean;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export interface FavoriteButtonProps {
  manualId: number;
  userId?: string;
  variant?: 'default' | 'outline' | 'ghost';
  size?: 'default' | 'sm' | 'lg' | 'icon';
}

export interface ExportMenuProps {
  manualId: number;
}

export interface RatingFormProps {
  manualId: number;
  userId?: string;
}

export interface RatingsListProps {
  ratings?: Rating[];
}

export interface ManualMetadataProps {
  manual: ManualDetailResponse;
}

export interface SearchInputProps {
  value: string;
  onChange: (value: string) => void;
  onSearch: () => void;
  placeholder?: string;
  disabled?: boolean;
  ariaLabel?: string;
}

export interface PageHeaderProps {
  title: string;
  description?: string;
  actions?: React.ReactNode;
  className?: string;
}

export interface PageContainerProps {
  children: React.ReactNode;
  className?: string;
}

export interface StatCardProps {
  title: string;
  value: string | number;
  description?: string;
  icon?: React.ComponentType<{ className?: string }>;
  className?: string;
  trend?: {
    value: number;
    isPositive: boolean;
  };
}

export interface EmptyStateProps {
  icon?: React.ComponentType<{ className?: string }>;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}

export interface LoadingStateProps {
  title?: string;
}

export interface ErrorStateProps {
  title: string;
  message?: string;
}

export interface PaginationProps {
  currentPage: number;
  totalItems: number;
  itemsPerPage: number;
  onPrevious: () => void;
  onNext: () => void;
  hasMore?: boolean;
}

export interface ActiveLinkProps {
  href: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  exact?: boolean;
}

