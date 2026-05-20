/**
 * Interface for decorative dot properties
 */
export interface DecorativeDot {
  id: number;
  left: string;
  top: string;
  delay: string;
  width: string;
  height: string;
  rotation: string;
  color: string;
  opacity: number;
  hasGlow: boolean;
}

/**
 * Navigation menu items configuration
 */
export interface NavItem {
  label: string;
  href?: string;
  onClick?: () => void;
  ariaLabel: string;
}


