"use client";
import { cn } from "../../utils/classNames";

type ErrorAlertProps = {
  readonly message: string;
  readonly className?: string;
};

export const ErrorAlert = ({ message, className }: ErrorAlertProps): JSX.Element => {
  return (
    <div
      className={cn(
        "mb-4 p-3 bg-red-100 text-red-700 rounded-lg text-sm",
        className
      )}
      role="alert"
      aria-live="assertive"
    >
      {message}
    </div>
  );
};



