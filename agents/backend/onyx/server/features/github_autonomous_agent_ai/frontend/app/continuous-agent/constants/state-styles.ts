/**
 * Shared styles for state components (loading, error, empty)
 */

export const STATE_STYLES = {
  CONTAINER: "flex items-center justify-center min-h-screen",
  CENTERED: "text-center py-16 border-2 border-dashed rounded-lg",
  LOADING: "text-lg",
  ERROR: "text-lg text-red-600",
  EMPTY_MESSAGE: "text-muted-foreground mb-4",
  BUTTON_PRIMARY: "px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2",
  BUTTON_SECONDARY: "px-4 py-2 text-sm text-blue-600 hover:text-blue-800 underline",
} as const;



