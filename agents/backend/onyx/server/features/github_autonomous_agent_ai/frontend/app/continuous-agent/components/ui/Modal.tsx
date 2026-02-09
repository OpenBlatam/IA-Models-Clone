"use client";

import { useCallback, useEffect, useRef, useMemo } from "react";
import type { ReactNode, MouseEvent, KeyboardEvent } from "react";
import { cn } from "../../utils/classNames";

type ModalProps = {
  readonly open: boolean;
  readonly onClose: () => void;
  readonly title: string;
  readonly children: ReactNode;
  readonly size?: "sm" | "md" | "lg" | "xl";
  readonly className?: string;
};

const SIZE_CLASSES = {
  sm: "max-w-md",
  md: "max-w-lg",
  lg: "max-w-2xl",
  xl: "max-w-4xl",
} as const;

/**
 * Modal component with accessibility features
 * 
 * Features:
 * - Keyboard navigation (Escape to close)
 * - Focus trap
 * - ARIA attributes
 * - Backdrop click to close
 * 
 * @param props - Component props
 * @returns The rendered modal component or null if closed
 */
export const Modal = ({
  open,
  onClose,
  title,
  children,
  size = "lg",
  className,
}: ModalProps): JSX.Element | null => {
  const modalRef = useRef<HTMLDivElement>(null);
  
  // Generate stable ID for accessibility
  const titleId = useMemo(
    () => `modal-title-${Math.random().toString(36).substring(7)}`,
    []
  );

  const handleBackdropClick = useCallback(
    (event: MouseEvent<HTMLDivElement>): void => {
      if (event.target === event.currentTarget) {
        onClose();
      }
    },
    [onClose]
  );

  const handleCloseKeyDown = useCallback(
    (event: KeyboardEvent<HTMLButtonElement>): void => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        onClose();
      }
    },
    [onClose]
  );

  useEffect(() => {
    if (!open) {
      return;
    }

    // Prevent body scroll when modal is open
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    // Handle escape key
    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    const keyDownHandler = handleKeyDown as unknown as EventListener;
    document.addEventListener("keydown", keyDownHandler);

    // Focus trap: focus the modal when it opens
    if (modalRef.current) {
      const focusableElements = modalRef.current.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      const firstElement = focusableElements[0] as HTMLElement;
      if (firstElement) {
        firstElement.focus();
      }
    }

    return () => {
      document.removeEventListener("keydown", keyDownHandler);
      document.body.style.overflow = originalOverflow;
    };
  }, [open, onClose]);

  if (!open) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      onClick={handleBackdropClick}
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
    >
      <div
        ref={modalRef}
        className={cn(
          "bg-background rounded-lg p-6 w-full max-h-[90vh] overflow-y-auto shadow-xl",
          SIZE_CLASSES[size],
          className
        )}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between items-center mb-6">
          <h2 id={titleId} className="text-2xl font-bold">
            {title}
          </h2>
          <button
            type="button"
            onClick={onClose}
            onKeyDown={handleCloseKeyDown}
            className="text-muted-foreground hover:text-foreground focus:outline-none focus:ring-2 focus:ring-blue-500 rounded p-1 transition-colors"
            aria-label="Cerrar modal"
            tabIndex={0}
          >
            <span aria-hidden="true" className="text-xl leading-none">×</span>
          </button>
        </div>
        {children}
      </div>
    </div>
  );
};

