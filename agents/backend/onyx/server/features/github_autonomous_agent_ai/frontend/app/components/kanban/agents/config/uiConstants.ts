/**
 * Constantes de UI centralizadas
 * Facilita el mantenimiento y consistencia visual
 */

export const UI_CLASSES = {
  container: "bg-white rounded-xl border border-gray-200 shadow-sm",
  header: "p-4 border-b border-gray-200",
  filters: "p-4 border-b border-gray-100 bg-gray-50/60",
  content: "p-4",
  card: {
    base: "p-3 rounded-lg border transition-all hover:shadow-sm",
    active: "bg-green-50 border-green-200",
    inactive: "bg-gray-50 border-gray-200",
  },
  table: {
    header: "bg-gray-50 border-b border-gray-200",
    row: "hover:bg-gray-50",
    cell: "px-3 py-2",
  },
} as const;

export const TEXT_SIZES = {
  xs: "text-xs",
  sm: "text-sm",
  "11px": "text-[11px]",
} as const;

export const SPACING = {
  gap1: "gap-1",
  gap2: "gap-2",
  gap3: "gap-3",
  spaceY3: "space-y-3",
} as const;








