import { useState, useMemo } from 'react';
import { useLocalStorage } from './useLocalStorage';
import { Task, TaskStatus } from '../types/task';

/**
 * Hook to manage core Kanban state (tasks, filters, selections)
 */
export function useKanbanState() {
  // Task selection and viewing
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);
  const [quickViewTask, setQuickViewTask] = useState<Task | null>(null);
  const [selectedTasks, setSelectedTasks] = useState<Set<string>>(new Set());
  const [comparisonMode, setComparisonMode] = useState<'date' | 'model' | 'repo' | null>(null);
  
  // Filters
  const [searchQuery, setSearchQuery] = useLocalStorage<string>('kanban-search', '');
  const [selectedRepository, setSelectedRepository] = useLocalStorage<string>('kanban-repo-filter', 'all');
  const [modelFilter, setModelFilter] = useLocalStorage<string>('kanban-model-filter', 'all');
  const [tagFilter, setTagFilter] = useState<string>('all');
  const [priorityFilter, setPriorityFilter] = useState<string>('all');
  const [errorFilter, setErrorFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<TaskStatus | 'all'>(() => {
    if (typeof window !== 'undefined') {
      return (localStorage.getItem('kanban-status-filter') as TaskStatus | 'all') || 'all';
    }
    return 'all';
  });
  const [dateFilter, setDateFilter] = useState<'all' | 'today' | 'week' | 'month' | 'custom'>(() => {
    if (typeof window !== 'undefined') {
      return (localStorage.getItem('kanban-date-filter') as 'all' | 'today' | 'week' | 'month' | 'custom') || 'all';
    }
    return 'all';
  });
  
  // Sorting
  const [sortBy, setSortBy] = useState<'date' | 'status' | 'repo' | 'model'>(() => {
    if (typeof window !== 'undefined') {
      return (localStorage.getItem('kanban-sort') as 'date' | 'status' | 'repo' | 'model') || 'date';
    }
    return 'date';
  });
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>(() => {
    if (typeof window !== 'undefined') {
      return (localStorage.getItem('kanban-sort-order') as 'asc' | 'desc') || 'desc';
    }
    return 'desc';
  });
  
  // View settings
  const [viewMode, setViewMode] = useLocalStorage<'kanban' | 'list' | 'timeline' | 'calendar'>('kanban-view-mode', 'kanban');
  const [groupByRepo, setGroupByRepo] = useLocalStorage<boolean>('kanban-group-repo', false);
  const [groupByModel, setGroupByModel] = useLocalStorage<boolean>('kanban-group-model', false);
  const [compactMode, setCompactMode] = useLocalStorage<boolean>('kanban-compact-mode', false);
  const [cardSize, setCardSize] = useState<'compact' | 'normal' | 'expanded'>(() => {
    if (typeof window !== 'undefined') {
      return (localStorage.getItem('kanban-card-size') as 'compact' | 'normal' | 'expanded') || 'normal';
    }
    return 'normal';
  });
  
  // Display settings
  const [showStats, setShowStats] = useLocalStorage<boolean>('kanban-stats', false);
  const [showMetrics, setShowMetrics] = useLocalStorage<boolean>('kanban-show-metrics', false);
  const [darkMode, setDarkMode] = useLocalStorage<boolean>('kanban-dark-mode', false);
  
  // Calendar view
  const [calendarMonth, setCalendarMonth] = useState<number | null>(null);
  const [calendarYear, setCalendarYear] = useState<number | null>(null);
  
  // Search settings
  const [searchInStreaming, setSearchInStreaming] = useState(false);
  const [showSearchSuggestions, setShowSearchSuggestions] = useState(false);
  
  // Clear all filters
  const clearAllFilters = () => {
    setSearchQuery('');
    setSelectedRepository('all');
    setModelFilter('all');
    setTagFilter('all');
    setPriorityFilter('all');
    setErrorFilter('all');
    setStatusFilter('all');
    setDateFilter('all');
  };

  return {
    // Task selection
    selectedTask,
    setSelectedTask,
    quickViewTask,
    setQuickViewTask,
    selectedTasks,
    setSelectedTasks,
    comparisonMode,
    setComparisonMode,
    
    // Filters
    searchQuery,
    setSearchQuery,
    selectedRepository,
    setSelectedRepository,
    modelFilter,
    setModelFilter,
    tagFilter,
    setTagFilter,
    priorityFilter,
    setPriorityFilter,
    errorFilter,
    setErrorFilter,
    statusFilter,
    setStatusFilter,
    dateFilter,
    setDateFilter,
    
    // Sorting
    sortBy,
    setSortBy,
    sortOrder,
    setSortOrder,
    
    // View settings
    viewMode,
    setViewMode,
    groupByRepo,
    setGroupByRepo,
    groupByModel,
    setGroupByModel,
    compactMode,
    setCompactMode,
    cardSize,
    setCardSize,
    
    // Display settings
    showStats,
    setShowStats,
    showMetrics,
    setShowMetrics,
    darkMode,
    setDarkMode,
    
    // Calendar
    calendarMonth,
    setCalendarMonth,
    calendarYear,
    setCalendarYear,
    
    // Search
    searchInStreaming,
    setSearchInStreaming,
    showSearchSuggestions,
    setShowSearchSuggestions,
    
    // Utilities
    clearAllFilters,
  };
}

