"use client";

import { cn } from "../../../../../utils/cn";

interface StatusBadgeProps {
  isActive: boolean;
  className?: string;
}

export function StatusBadge({ isActive, className }: StatusBadgeProps) {
  return (
    <span
      className={cn(
        "px-2 py-0.5 text-xs font-medium rounded-full",
        isActive ? "bg-green-100 text-green-700" : "bg-gray-100 text-gray-700",
        className
      )}
    >
      {isActive ? "Activo" : "Inactivo"}
    </span>
  );
}








