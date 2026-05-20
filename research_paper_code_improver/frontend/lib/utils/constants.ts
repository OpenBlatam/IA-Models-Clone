/**
 * Application constants
 */

export const constants = {
  API_BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8030',
  
  STORAGE_KEYS: {
    CODE_HISTORY: 'code-history',
    PREFERENCES: 'preferences',
    RECENT_PAPERS: 'recent-papers',
    FILTERS: 'filters',
  },

  FILE_UPLOAD: {
    MAX_SIZE_MB: 50,
    ALLOWED_TYPES: ['application/pdf'],
  },

  DEBOUNCE_DELAY: {
    SEARCH: 300,
    INPUT: 500,
  },

  PAGINATION: {
    DEFAULT_PAGE_SIZE: 20,
    MAX_PAGE_SIZE: 100,
  },

  HISTORY: {
    MAX_ITEMS: 50,
  },

  LANGUAGES: [
    { value: 'python', label: 'Python', extension: '.py' },
    { value: 'javascript', label: 'JavaScript', extension: '.js' },
    { value: 'typescript', label: 'TypeScript', extension: '.ts' },
    { value: 'java', label: 'Java', extension: '.java' },
    { value: 'cpp', label: 'C++', extension: '.cpp' },
    { value: 'c', label: 'C', extension: '.c' },
    { value: 'go', label: 'Go', extension: '.go' },
    { value: 'rust', label: 'Rust', extension: '.rs' },
    { value: 'php', label: 'PHP', extension: '.php' },
    { value: 'ruby', label: 'Ruby', extension: '.rb' },
  ],

  COLORS: {
    PRIMARY: '#667eea',
    SUCCESS: '#10b981',
    WARNING: '#f59e0b',
    ERROR: '#ef4444',
    INFO: '#3b82f6',
  },
}




