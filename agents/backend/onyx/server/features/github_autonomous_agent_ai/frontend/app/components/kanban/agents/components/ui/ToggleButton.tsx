"use client";

import { cn } from "../../../../../utils/cn";

interface ToggleButtonProps {
  isActive: boolean;
  onClick: () => void;
  className?: string;
  size?: "sm" | "md";
}

export function ToggleButton({
  isActive,
  onClick,
  className,
  size = "md",
}: ToggleButtonProps) {
  const sizeClasses = {
    sm: "px-2 py-1 text-xs",
    md: "px-2.5 py-1 text-xs",
  };

  return (
    <button
      onClick={onClick}
      className={cn(
        "font-medium rounded-full border transition-colors",
        sizeClasses[size],
        isActive
          ? "border-red-200 text-red-700 bg-red-50 hover:bg-red-100"
          : "border-green-200 text-green-700 bg-green-50 hover:bg-green-100",
        className
      )}
    >
      {isActive ? "Pausar" : "Activar"}
    </button>
  );
}








