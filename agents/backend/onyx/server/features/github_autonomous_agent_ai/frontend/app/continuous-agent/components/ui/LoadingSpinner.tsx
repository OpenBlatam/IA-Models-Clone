"use client";
import { cn } from "../../utils/classNames";

type LoadingSpinnerProps = {
  readonly size?: "sm" | "md" | "lg";
  readonly className?: string;
  readonly label?: string;
};

const SIZE_CLASSES = {
  sm: "w-4 h-4",
  md: "w-6 h-6",
  lg: "w-8 h-8",
} as const;

export const LoadingSpinner = ({
  size = "md",
  className,
  label = "Cargando...",
}: LoadingSpinnerProps): JSX.Element => {
  return (
    <div className={cn("flex items-center justify-center gap-2", className)}>
      <div
        className={cn(
          "animate-spin rounded-full border-2 border-gray-300 border-t-blue-600",
          SIZE_CLASSES[size]
        )}
        role="status"
        aria-label={label}
      >
        <span className="sr-only">{label}</span>
      </div>
    </div>
  );
};



