"use client";

import React, { useCallback, memo, useMemo } from "react";

type ToggleSize = "sm" | "md" | "lg";

type ToggleSwitchProps = {
  readonly checked: boolean;
  readonly onChange: (checked: boolean) => void;
  readonly disabled?: boolean;
  readonly label?: string;
  readonly size?: ToggleSize;
  readonly ariaLabel?: string;
};

const SIZE_CLASSES: Record<ToggleSize, string> = {
  sm: "w-8 h-4",
  md: "w-11 h-6",
  lg: "w-14 h-7",
};

const DOT_SIZE_CLASSES: Record<ToggleSize, string> = {
  sm: "w-3 h-3",
  md: "w-5 h-5",
  lg: "w-6 h-6",
};

const TRANSLATE_CLASSES: Record<ToggleSize, { checked: string; unchecked: string }> = {
  sm: { checked: "translate-x-4", unchecked: "translate-x-0" },
  md: { checked: "translate-x-5", unchecked: "translate-x-0" },
  lg: { checked: "translate-x-7", unchecked: "translate-x-0" },
};

/**
 * Accessible toggle switch component
 * 
 * Features:
 * - Keyboard navigation (Enter/Space to toggle)
 * - ARIA attributes for screen readers
 * - Multiple size options
 * - Disabled state support
 * 
 * @param props - Component props
 * @returns The rendered toggle switch component
 */
const ToggleSwitchComponent = ({
  checked,
  onChange,
  disabled = false,
  label,
  size = "md",
  ariaLabel,
}: ToggleSwitchProps): JSX.Element => {
  const handleClick = useCallback((): void => {
    if (!disabled) {
      onChange(!checked);
    }
  }, [disabled, checked, onChange]);

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLButtonElement>): void => {
      if (disabled) {
        return;
      }

      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        onChange(!checked);
      }
    },
    [disabled, checked, onChange]
  );

  // Memoize computed classes to prevent recalculation
  const sizeClass = useMemo(() => SIZE_CLASSES[size], [size]);
  const dotSizeClass = useMemo(() => DOT_SIZE_CLASSES[size], [size]);
  const translateClass = useMemo(
    () => (checked ? TRANSLATE_CLASSES[size].checked : TRANSLATE_CLASSES[size].unchecked),
    [checked, size]
  );
  const backgroundColorClass = useMemo(
    () => (checked ? "bg-green-500" : "bg-gray-300"),
    [checked]
  );
  const disabledClass = useMemo(
    () => (disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"),
    [disabled]
  );
  const labelClass = useMemo(
    () => (disabled ? "text-muted-foreground" : "text-foreground"),
    [disabled]
  );

  const computedAriaLabel = useMemo(
    () => ariaLabel || label || (checked ? "Activo" : "Inactivo"),
    [ariaLabel, label, checked]
  );

  return (
    <div className="flex items-center gap-3">
      {label && (
        <span className={`text-sm font-medium ${labelClass}`} aria-hidden="true">
          {label}
        </span>
      )}
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={computedAriaLabel}
        tabIndex={disabled ? -1 : 0}
        disabled={disabled}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        className={`relative inline-flex items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${sizeClass} ${backgroundColorClass} ${disabledClass}`}
      >
        <span
          className={`inline-block rounded-full bg-white shadow-lg transform transition-transform ${dotSizeClass} ${translateClass}`}
          aria-hidden="true"
        />
      </button>
    </div>
  );
};

// Memoize component to prevent unnecessary re-renders
export const ToggleSwitch = memo(ToggleSwitchComponent, (prevProps, nextProps) => {
  return (
    prevProps.checked === nextProps.checked &&
    prevProps.disabled === nextProps.disabled &&
    prevProps.size === nextProps.size &&
    prevProps.label === nextProps.label &&
    prevProps.ariaLabel === nextProps.ariaLabel &&
    prevProps.onChange === nextProps.onChange
  );
});

ToggleSwitch.displayName = "ToggleSwitch";

