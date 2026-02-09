"use client";

import type { ViewMode } from "../../types";

interface ViewModeToggleProps {
  currentMode: ViewMode;
  onToggle: () => void;
  className?: string;
}

export function ViewModeToggle({
  currentMode,
  onToggle,
  className,
}: ViewModeToggleProps) {
  return (
    <button
      onClick={onToggle}
      className={`inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border text-xs font-medium text-gray-700 hover:bg-gray-50 ${className}`}
    >
      {currentMode === "table" ? "Vista tarjetas" : "Vista tabla"}
    </button>
  );
}








