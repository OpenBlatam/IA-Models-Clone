"use client";

import React from "react";
import { cn } from "../../utils/classNames";

type CharacterCountStatus = "ok" | "warning" | "error";

type CharacterCountProps = {
  readonly current: number;
  readonly max: number;
  readonly min?: number;
  readonly status?: CharacterCountStatus;
  readonly className?: string;
  readonly id?: string;
};

export const CharacterCount = ({
  current,
  max,
  min,
  status = "ok",
  className,
  id,
}: CharacterCountProps): JSX.Element => {
  const getStatusClass = (): string => {
    switch (status) {
      case "error":
        return "text-red-500";
      case "warning":
        return "text-yellow-600";
      default:
        return "text-gray-400";
    }
  };

  const percentage = (current / max) * 100;
  const isNearLimit = percentage > 80;
  const isOverLimit = current > max;

  return (
    <div
      id={id}
      className={cn(
        "text-xs pointer-events-none",
        getStatusClass(),
        className
      )}
      aria-live="polite"
      aria-atomic="true"
    >
      <span className="font-medium">{current}</span>
      <span className="text-gray-400">/{max}</span>
      {min && current < min && (
        <span className="ml-1 text-yellow-600">(mín. {min})</span>
      )}
      {isOverLimit && (
        <span className="ml-1 text-red-500 font-semibold">(excedido)</span>
      )}
      {isNearLimit && !isOverLimit && (
        <span className="ml-1 text-yellow-600">(cerca del límite)</span>
      )}
    </div>
  );
};







