/**
 * Centralized TypeScript types and interfaces
 * 
 * Re-exporta todos los tipos desde archivos específicos
 */

// Tipos de mensajes
export * from './message.types'

// Tipos de estados
export * from './state.types'

// Tipos legacy (compatibilidad)
import { Model } from '@/store/modelStore'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
}

export interface ChatProps {
  devMode?: boolean
}

export interface ModelPreview {
  spec: any
  validation: {
    valid: boolean
    errors: string[]
  }
}

export interface ChatContextValue {
  messages: Message[]
  isLoading: boolean
  selectedModels: Model[]
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => void
  clearMessages: () => void
}

// Message Types
export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  metadata?: MessageMetadata
}

export interface MessageMetadata {
  wordCount?: number
  charCount?: number
  hasCode?: boolean
  hasLinks?: boolean
  links?: string[]
  language?: string
  sentiment?: 'positive' | 'negative' | 'neutral'
  [key: string]: any
}

// Chat Types
export interface ChatState {
  input: string
  isLoading: boolean
  messages: Message[]
  showPreview: boolean
  showHistory: boolean
  showComparator: boolean
  selectedModels: any[]
  validation: ValidationResult | null
  previewSpec: any
  modelHistory: any[]
}

export interface ChatUIState {
  showTour: boolean
  showMetrics: boolean
  showProactive: boolean
  showStats: boolean
  viewMode: 'normal' | 'compact' | 'comfortable'
  theme: 'dark' | 'light' | 'auto'
  fontSize: 'small' | 'medium' | 'large'
  autoScroll: boolean
}

export interface ChatFeatureFlags {
  useBulkChatMode: boolean
  voiceInputEnabled: boolean
  voiceOutputEnabled: boolean
  readMode: boolean
  presentationMode: boolean
  collaborationMode: boolean
  devMode: boolean
  accessibilityMode: boolean
}

// Validation Types
export interface ValidationResult {
  valid: boolean
  errors: string[]
  warnings?: string[]
}

// Search Types
export interface SearchFilters {
  dateRange?: { start: Date; end: Date }
  minWords?: number
  maxWords?: number
  hasCode?: boolean
  hasLinks?: boolean
  role?: 'all' | 'user' | 'assistant' | 'system'
}

export interface SearchResult {
  messageId: string
  matches: number[]
  score: number
}

// Export/Import Types
export type ExportFormat = 'json' | 'txt' | 'md' | 'html' | 'csv' | 'xml' | 'yaml' | 'pdf'

export interface ExportOptions {
  format: ExportFormat
  includeMetadata?: boolean
  includeTimestamps?: boolean
  filter?: (message: Message) => boolean
}

export interface ImportResult {
  success: boolean
  messages: Message[]
  errors?: string[]
  warnings?: string[]
}

// Theme Types
export interface ThemeConfig {
  background: string
  foreground: string
  primary: string
  secondary: string
  accent: string
  border: string
  [key: string]: string
}

export interface ThemePreset {
  name: string
  config: ThemeConfig
  colors?: Map<string, string>
}

// Notification Types
export interface NotificationRule {
  keywords: string[]
  enabled: boolean
  sound?: boolean
  desktop?: boolean
}

export interface NotificationOptions {
  body?: string
  icon?: string
  badge?: string
  tag?: string
  silent?: boolean
  requireInteraction?: boolean
}

// Collaboration Types
export interface Collaborator {
  id: string
  name: string
  avatar?: string
  role?: 'viewer' | 'editor' | 'admin'
}

export interface MessageComment {
  id: string
  author: string
  content: string
  timestamp: number
  edited?: boolean
  editedAt?: number
}

export interface MessageVote {
  up: number
  down: number
  userVote?: 'up' | 'down'
}

// Performance Types
export interface PerformanceMetrics {
  renderCount: number
  averageRenderTime: number
  cacheSize: number
  cacheHitRate: number
  totalRenders: number
  [key: string]: number
}

export interface CacheEntry<T> {
  content: T
  timestamp: number
  ttl?: number
}

// Voice Types
export interface VoiceRecording {
  blob: Blob
  duration: number
  format: string
  transcription?: string
}

export interface VoiceSettings {
  language: string
  continuous: boolean
  interimResults: boolean
  maxAlternatives: number
}

// Storage Types
export interface StorageOptions {
  ttl?: number
  compress?: boolean
  encrypt?: boolean
}

export interface StorageEntry<T> {
  value: T
  expires?: number
  metadata?: {
    createdAt: number
    updatedAt: number
    version?: number
  }
}

// Component Props Types
export interface MessageListProps {
  messages: Message[]
  viewMode?: 'normal' | 'compact' | 'comfortable'
  filterRole?: 'all' | 'user' | 'assistant' | 'system'
  searchQuery?: string
  highlightSearch?: boolean
  showReactions?: boolean
  showTags?: boolean
  showTimestamps?: boolean
  onMessageClick?: (messageId: string) => void
  onMessageAction?: (messageId: string, action: string) => void
}

export interface InputAreaProps {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  isLoading?: boolean
  placeholder?: string
  voiceInputEnabled?: boolean
  voiceOutputEnabled?: boolean
  showQuickActions?: boolean
  maxLength?: number
  autoFocus?: boolean
  disabled?: boolean
}

export interface ToolbarProps {
  searchQuery: string
  onSearchChange: (query: string) => void
  showFilters: boolean
  onToggleFilters: () => void
  viewMode: 'normal' | 'compact' | 'comfortable'
  onViewModeChange: (mode: 'normal' | 'compact' | 'comfortable') => void
  onExport?: () => void
  onImport?: () => void
  onShare?: () => void
  onSettings?: () => void
  onNotifications?: () => void
  showSearch?: boolean
  showFiltersButton?: boolean
  showViewControls?: boolean
  showActions?: boolean
}

export interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  activePanel?: 'history' | 'bookmarks' | 'settings' | 'archive' | 'tags'
  onPanelChange?: (panel: string | null) => void
  children?: React.ReactNode
}

// Modal Types
export interface BaseModalProps {
  isOpen: boolean
  onClose: () => void
  title: string
  children: React.ReactNode
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full'
  showCloseButton?: boolean
}

export interface ConfirmDialogProps {
  isOpen: boolean
  onClose: () => void
  onConfirm: () => void
  title: string
  message: string
  confirmText?: string
  cancelText?: string
  variant?: 'danger' | 'warning' | 'info'
}

// Hook Return Types
export interface UseChatStateReturn {
  state: ChatState
  uiState: ChatUIState
  featureFlags: ChatFeatureFlags
  updateState: (updates: Partial<ChatState>) => void
  updateUIState: (updates: Partial<ChatUIState>) => void
  updateFeatureFlags: (updates: Partial<ChatFeatureFlags>) => void
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => void
  clearMessages: () => void
}

export interface UseMessageManagementReturn {
  favoriteMessages: Set<string>
  pinnedMessages: Set<string>
  archivedMessages: Set<string>
  selectedMessages: Set<string>
  messageTags: Map<string, string[]>
  messageNotes: Map<string, string>
  messageReactions: Map<string, string[]>
  messageBookmarks: Map<string, { name: string, category: string, tags: string[] }>
  toggleFavorite: (messageId: string) => void
  togglePin: (messageId: string) => void
  toggleArchive: (messageId: string) => void
  toggleSelection: (messageId: string) => void
  addTag: (messageId: string, tag: string) => void
  removeTag: (messageId: string, tag: string) => void
  addNote: (messageId: string, note: string) => void
  removeNote: (messageId: string) => void
  addReaction: (messageId: string, reaction: string) => void
  removeReaction: (messageId: string, reaction: string) => void
  addBookmark: (messageId: string, name: string, category?: string, tags?: string[]) => void
  removeBookmark: (messageId: string) => void
}

export interface UseSearchAndFiltersReturn {
  searchQuery: string
  currentSearchIndex: number
  filteredMessages: Message[]
  searchFilters: SearchFilters
  advancedSearch: boolean
  highlightSearch: boolean
  searchIndex: Map<string, number[]>
  setSearchQuery: (query: string) => void
  setCurrentSearchIndex: (index: number) => void
  setSearchFilters: (filters: SearchFilters) => void
  setAdvancedSearch: (enabled: boolean) => void
  setHighlightSearch: (enabled: boolean) => void
  nextMatch: () => void
  previousMatch: () => void
  clearSearch: () => void
  updateFilter: <K extends keyof SearchFilters>(key: K, value: SearchFilters[K]) => void
}

// Event Types
export interface ChatEvent {
  type: 'message_sent' | 'message_received' | 'error' | 'loading' | 'export' | 'import'
  timestamp: number
  data?: any
}

export interface MessageEvent extends ChatEvent {
  type: 'message_sent' | 'message_received'
  data: {
    messageId: string
    role: 'user' | 'assistant'
    content: string
  }
}

// Error Types
export interface ChatError {
  code: string
  message: string
  details?: any
  timestamp: number
}

export class ChatInterfaceError extends Error {
  constructor(
    public code: string,
    message: string,
    public details?: any
  ) {
    super(message)
    this.name = 'ChatInterfaceError'
  }
}

// Utility Types
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P]
}

export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>

export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>

// React Types
import React from 'react'

export type ReactNode = React.ReactNode
export type ReactElement = React.ReactElement
export type ComponentType<P = {}> = React.ComponentType<P>


