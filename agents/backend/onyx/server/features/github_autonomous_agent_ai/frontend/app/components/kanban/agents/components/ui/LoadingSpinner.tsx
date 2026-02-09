"use client";

interface LoadingSpinnerProps {
  message?: string;
  className?: string;
}

export function LoadingSpinner({
  message = "Cargando...",
  className,
}: LoadingSpinnerProps) {
  return (
    <div className={`py-8 text-center text-sm text-gray-500 ${className}`}>
      {message}
    </div>
  );
}








