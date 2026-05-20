/**
 * Shared type definitions for component props and common patterns
 */

import type { ReactNode } from "react";

/**
 * Base props for components that accept children
 */
export type ComponentWithChildren = {
  readonly children: ReactNode;
};

/**
 * Base props for components with className
 */
export type ComponentWithClassName = {
  readonly className?: string;
};

/**
 * Base props for components with optional id
 */
export type ComponentWithId = {
  readonly id?: string;
};

/**
 * Base props for form field components
 */
export type FormFieldProps = ComponentWithClassName & {
  readonly label?: string;
  readonly error?: string | null;
  readonly required?: boolean;
  readonly helpText?: string;
};

/**
 * Base props for modal components
 */
export type ModalProps = {
  readonly open: boolean;
  readonly onClose: () => void;
};

/**
 * Base props for components that handle async operations
 */
export type AsyncOperationProps = {
  readonly isLoading?: boolean;
  readonly error?: string | null;
  readonly onRetry?: () => void;
};

/**
 * Base props for components with loading state
 */
export type LoadingProps = {
  readonly isLoading?: boolean;
  readonly loadingText?: string;
};

/**
 * Base props for components with error state
 */
export type ErrorProps = {
  readonly error?: string | null;
  readonly onDismiss?: () => void;
};



