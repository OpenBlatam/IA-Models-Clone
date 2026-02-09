"use client";

import { cn } from "../../../../../utils/cn";

interface ExpandButtonProps {
  isExpanded: boolean;
  onClick: () => void;
  className?: string;
}

export function ExpandButton({ isExpanded, onClick, className }: ExpandButtonProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "p-1 rounded-full hover:bg-gray-100 transition-colors",
        className
      )}
    >
      <svg
        className={cn(
          "w-4 h-4 text-gray-600 transition-transform",
          isExpanded ? "rotate-90" : ""
        )}
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9 5l7 7-7 7"
        />
      </svg>
    </button>
  );
}








