"use client";

import { cn } from "../../../../../utils/cn";

interface StatusIndicatorProps {
  isActive: boolean;
  className?: string;
}

export function StatusIndicator({ isActive, className }: StatusIndicatorProps) {
  return (
    <span
      className={cn(
        "w-2 h-2 rounded-full flex-shrink-0",
        isActive ? "bg-green-500 animate-pulse" : "bg-gray-400",
        className
      )}
    />
  );
}








