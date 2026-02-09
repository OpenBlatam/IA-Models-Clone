/**
 * Shared style constants for consistent styling across components
 * 
 * Provides reusable className strings and style patterns
 */

/**
 * Common card styles
 */
export const CARD_STYLES = {
  BASE: "bg-card border rounded-lg p-6 hover:shadow-lg transition-shadow",
  HEADER: "flex justify-between items-start mb-4",
  CONTENT: "space-y-2 mb-4",
  ACTIONS: "flex gap-2 mt-4",
  STATS_ROW: "flex justify-between text-sm",
} as const;

/**
 * Common text styles
 */
export const TEXT_STYLES = {
  TITLE: "text-xl font-semibold mb-1",
  SUBTITLE: "text-sm text-muted-foreground",
  LABEL: "text-muted-foreground",
  VALUE: "font-medium",
  SMALL: "text-xs text-muted-foreground",
} as const;

/**
 * Status indicator styles
 */
export const STATUS_STYLES = {
  ACTIVE: "text-green-600",
  INACTIVE: "text-gray-600",
  NEXT_EXECUTION_ACTIVE: "text-blue-600",
  SUCCESS: "text-green-600 font-medium",
  ERROR: "text-red-600",
  WARNING: "text-yellow-600",
} as const;

/**
 * Status text constants
 */
export const STATUS_TEXT = {
  ACTIVE: "🟢 Activo",
  INACTIVE: "⚫ Inactivo",
} as const;

/**
 * Badge/variant styles
 */
export const BADGE_STYLES = {
  SUCCESS: "bg-green-100 text-green-700",
  ERROR: "bg-red-100 text-red-700",
  WARNING: "bg-yellow-100 text-yellow-700",
  BASE: "inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium",
} as const;

/**
 * Modal styles
 */
export const MODAL_STYLES = {
  OVERLAY: "fixed inset-0 bg-black/50 z-50",
  CONTENT: "fixed left-[50%] top-[50%] z-50 grid w-full max-w-2xl translate-x-[-50%] translate-y-[-50%] gap-4 border bg-background p-6 shadow-lg duration-200 rounded-lg max-h-[90vh] overflow-y-auto",
  TITLE: "text-2xl font-bold mb-4",
} as const;

/**
 * Form styles
 */
export const FORM_STYLES = {
  CONTAINER: "space-y-4",
  FOOTER: "flex justify-end gap-2 pt-4 border-t",
  FIELD_CONTAINER: "relative",
} as const;

/**
 * Layout styles
 */
export const LAYOUT_STYLES = {
  FLEX_BETWEEN: "flex justify-between items-center",
  FLEX_START: "flex justify-between items-start",
  GRID_CARDS: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6",
} as const;



