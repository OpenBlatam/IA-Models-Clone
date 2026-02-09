/**
 * Accessibility utilities for ARIA attributes
 * 
 * Provides helper functions for consistent ARIA attribute management
 */

/**
 * ARIA live region politeness levels
 */
export type AriaLivePoliteness = "off" | "polite" | "assertive";

/**
 * ARIA busy state
 */
export interface AriaBusyState {
  readonly "aria-busy": boolean;
  readonly "aria-live"?: AriaLivePoliteness;
}

/**
 * Creates ARIA busy attributes
 */
export function createAriaBusy(busy: boolean, live: AriaLivePoliteness = "polite"): AriaBusyState {
  return {
    "aria-busy": busy,
    "aria-live": busy ? live : undefined,
  };
}

/**
 * ARIA label attributes
 */
export interface AriaLabelAttributes {
  readonly "aria-label": string;
  readonly "aria-labelledby"?: string;
  readonly "aria-describedby"?: string;
}

/**
 * Creates ARIA label attributes
 */
export function createAriaLabel(
  label: string,
  options?: {
    readonly labelledBy?: string;
    readonly describedBy?: string;
  }
): AriaLabelAttributes {
  return {
    "aria-label": label,
    ...(options?.labelledBy && { "aria-labelledby": options.labelledBy }),
    ...(options?.describedBy && { "aria-describedby": options.describedBy }),
  };
}

/**
 * ARIA live region attributes
 */
export interface AriaLiveAttributes {
  readonly "aria-live": AriaLivePoliteness;
  readonly "aria-atomic"?: boolean;
}

/**
 * Creates ARIA live region attributes
 */
export function createAriaLive(
  politeness: AriaLivePoliteness = "polite",
  atomic: boolean = true
): AriaLiveAttributes {
  return {
    "aria-live": politeness,
    "aria-atomic": atomic,
  };
}

/**
 * ARIA expanded state
 */
export interface AriaExpandedState {
  readonly "aria-expanded": boolean;
  readonly "aria-controls"?: string;
}

/**
 * Creates ARIA expanded attributes
 */
export function createAriaExpanded(
  expanded: boolean,
  controls?: string
): AriaExpandedState {
  return {
    "aria-expanded": expanded,
    ...(controls && { "aria-controls": controls }),
  };
}

/**
 * ARIA disabled state
 */
export interface AriaDisabledState {
  readonly "aria-disabled": boolean;
}

/**
 * Creates ARIA disabled attributes
 */
export function createAriaDisabled(disabled: boolean): AriaDisabledState {
  return {
    "aria-disabled": disabled,
  };
}

/**
 * ARIA invalid state
 */
export interface AriaInvalidState {
  readonly "aria-invalid": boolean | "grammar" | "spelling";
  readonly "aria-errormessage"?: string;
}

/**
 * Creates ARIA invalid attributes
 */
export function createAriaInvalid(
  invalid: boolean | "grammar" | "spelling",
  errorMessageId?: string
): AriaInvalidState {
  return {
    "aria-invalid": invalid,
    ...(errorMessageId && { "aria-errormessage": errorMessageId }),
  };
}

/**
 * ARIA role and state attributes
 */
export interface AriaRoleAttributes {
  readonly role?: string;
  readonly "aria-roledescription"?: string;
}

/**
 * Creates ARIA role attributes
 */
export function createAriaRole(
  role: string,
  description?: string
): AriaRoleAttributes {
  return {
    role,
    ...(description && { "aria-roledescription": description }),
  };
}

/**
 * Combines multiple ARIA attribute objects
 */
export function combineAriaAttributes(
  ...attributes: Array<Record<string, unknown>>
): Record<string, unknown> {
  return Object.assign({}, ...attributes);
}




