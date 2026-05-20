/**
 * Keyboard navigation utilities
 * 
 * Provides helper functions for keyboard event handling and accessibility
 */

/**
 * Common keyboard keys
 */
export const KEYBOARD_KEYS = {
  ENTER: "Enter",
  SPACE: " ",
  ESCAPE: "Escape",
  TAB: "Tab",
  ARROW_UP: "ArrowUp",
  ARROW_DOWN: "ArrowDown",
  ARROW_LEFT: "ArrowLeft",
  ARROW_RIGHT: "ArrowRight",
  HOME: "Home",
  END: "End",
  PAGE_UP: "PageUp",
  PAGE_DOWN: "PageDown",
} as const;

/**
 * Checks if a key is an activation key (Enter or Space)
 */
export function isActivationKey(key: string): boolean {
  return key === KEYBOARD_KEYS.ENTER || key === KEYBOARD_KEYS.SPACE;
}

/**
 * Checks if a key is an arrow key
 */
export function isArrowKey(key: string): boolean {
  return [
    KEYBOARD_KEYS.ARROW_UP,
    KEYBOARD_KEYS.ARROW_DOWN,
    KEYBOARD_KEYS.ARROW_LEFT,
    KEYBOARD_KEYS.ARROW_RIGHT,
  ].includes(key as typeof KEYBOARD_KEYS[keyof typeof KEYBOARD_KEYS]);
}

/**
 * Checks if a key is a navigation key
 */
export function isNavigationKey(key: string): boolean {
  return (
    isArrowKey(key) ||
    key === KEYBOARD_KEYS.HOME ||
    key === KEYBOARD_KEYS.END ||
    key === KEYBOARD_KEYS.PAGE_UP ||
    key === KEYBOARD_KEYS.PAGE_DOWN
  );
}

/**
 * Checks if modifier keys are pressed
 */
export interface ModifierKeys {
  readonly ctrl: boolean;
  readonly shift: boolean;
  readonly alt: boolean;
  readonly meta: boolean;
}

/**
 * Gets modifier keys state from keyboard event
 */
export function getModifierKeys(event: KeyboardEvent): ModifierKeys {
  return {
    ctrl: event.ctrlKey,
    shift: event.shiftKey,
    alt: event.altKey,
    meta: event.metaKey,
  };
}

/**
 * Checks if Ctrl+Enter is pressed
 */
export function isCtrlEnter(event: KeyboardEvent): boolean {
  return event.ctrlKey && event.key === KEYBOARD_KEYS.ENTER;
}

/**
 * Checks if Ctrl+Shift+F is pressed
 */
export function isCtrlShiftF(event: KeyboardEvent): boolean {
  return event.ctrlKey && event.shiftKey && event.key === "f";
}

/**
 * Checks if Escape is pressed
 */
export function isEscape(event: KeyboardEvent): boolean {
  return event.key === KEYBOARD_KEYS.ESCAPE;
}

/**
 * Prevents default behavior and stops propagation
 */
export function preventDefault(event: React.KeyboardEvent | KeyboardEvent): void {
  event.preventDefault();
  event.stopPropagation();
}

/**
 * Creates a keyboard event handler
 */
export function createKeyboardHandler(
  handlers: {
    readonly onEnter?: () => void;
    readonly onSpace?: () => void;
    readonly onEscape?: () => void;
    readonly onArrowUp?: () => void;
    readonly onArrowDown?: () => void;
    readonly onArrowLeft?: () => void;
    readonly onArrowRight?: () => void;
    readonly onHome?: () => void;
    readonly onEnd?: () => void;
    readonly onCtrlEnter?: () => void;
    readonly onCtrlShiftF?: () => void;
    readonly onAny?: (key: string) => void;
  }
): (event: React.KeyboardEvent) => void {
  return (event: React.KeyboardEvent) => {
    const key = event.key;

    switch (key) {
      case KEYBOARD_KEYS.ENTER:
        if (event.ctrlKey && handlers.onCtrlEnter) {
          handlers.onCtrlEnter();
          preventDefault(event);
        } else if (handlers.onEnter) {
          handlers.onEnter();
          preventDefault(event);
        }
        break;
      case KEYBOARD_KEYS.SPACE:
        if (handlers.onSpace) {
          handlers.onSpace();
          preventDefault(event);
        }
        break;
      case KEYBOARD_KEYS.ESCAPE:
        if (handlers.onEscape) {
          handlers.onEscape();
          preventDefault(event);
        }
        break;
      case KEYBOARD_KEYS.ARROW_UP:
        if (handlers.onArrowUp) {
          handlers.onArrowUp();
          preventDefault(event);
        }
        break;
      case KEYBOARD_KEYS.ARROW_DOWN:
        if (handlers.onArrowDown) {
          handlers.onArrowDown();
          preventDefault(event);
        }
        break;
      case KEYBOARD_KEYS.ARROW_LEFT:
        if (handlers.onArrowLeft) {
          handlers.onArrowLeft();
          preventDefault(event);
        }
        break;
      case KEYBOARD_KEYS.ARROW_RIGHT:
        if (handlers.onArrowRight) {
          handlers.onArrowRight();
          preventDefault(event);
        }
        break;
      case KEYBOARD_KEYS.HOME:
        if (handlers.onHome) {
          handlers.onHome();
          preventDefault(event);
        }
        break;
      case KEYBOARD_KEYS.END:
        if (handlers.onEnd) {
          handlers.onEnd();
          preventDefault(event);
        }
        break;
      default:
        if (event.ctrlKey && event.shiftKey && key === "f" && handlers.onCtrlShiftF) {
          handlers.onCtrlShiftF();
          preventDefault(event);
        }
        break;
    }

    if (handlers.onAny) {
      handlers.onAny(key);
    }
  };
}




