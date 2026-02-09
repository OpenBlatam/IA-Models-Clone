/**
 * Advanced hotkeys system
 * @module robot-3d-view/utils/hotkeys-advanced
 */

/**
 * Hotkey combination
 */
export interface HotkeyCombination {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  meta?: boolean;
}

/**
 * Hotkey handler
 */
export type HotkeyHandler = (event: KeyboardEvent) => void | boolean;

/**
 * Hotkey registration
 */
export interface HotkeyRegistration {
  id: string;
  combination: HotkeyCombination;
  handler: HotkeyHandler;
  description?: string;
  enabled?: boolean;
  preventDefault?: boolean;
  stopPropagation?: boolean;
}

/**
 * Advanced Hotkeys Manager class
 */
export class AdvancedHotkeysManager {
  private hotkeys: Map<string, HotkeyRegistration> = new Map();
  private enabled = true;
  private scope: HTMLElement | Window = window;

  /**
   * Registers a hotkey
   */
  register(registration: HotkeyRegistration): () => void {
    this.hotkeys.set(registration.id, {
      ...registration,
      enabled: registration.enabled !== false,
      preventDefault: registration.preventDefault !== false,
      stopPropagation: registration.stopPropagation || false,
    });

    // Return unsubscribe function
    return () => {
      this.hotkeys.delete(registration.id);
    };
  }

  /**
   * Unregisters a hotkey
   */
  unregister(id: string): boolean {
    return this.hotkeys.delete(id);
  }

  /**
   * Enables/disables a hotkey
   */
  setHotkeyEnabled(id: string, enabled: boolean): boolean {
    const hotkey = this.hotkeys.get(id);
    if (!hotkey) return false;

    hotkey.enabled = enabled;
    return true;
  }

  /**
   * Enables/disables all hotkeys
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  /**
   * Sets hotkey scope
   */
  setScope(scope: HTMLElement | Window): void {
    this.scope = scope;
  }

  /**
   * Checks if a keyboard event matches a combination
   */
  private matchesCombination(
    event: KeyboardEvent,
    combination: HotkeyCombination
  ): boolean {
    const keyMatches = event.key.toLowerCase() === combination.key.toLowerCase();
    const ctrlMatches = combination.ctrl ? event.ctrlKey : !event.ctrlKey;
    const shiftMatches = combination.shift ? event.shiftKey : !event.shiftKey;
    const altMatches = combination.alt ? event.altKey : !event.altKey;
    const metaMatches = combination.meta ? event.metaKey : !event.metaKey;

    return keyMatches && ctrlMatches && shiftMatches && altMatches && metaMatches;
  }

  /**
   * Handles keyboard events
   */
  private handleKeyDown = (event: KeyboardEvent): void => {
    if (!this.enabled) return;

    // Ignore if typing in input
    const target = event.target as HTMLElement;
    if (
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target?.isContentEditable
    ) {
      return;
    }

    // Check all hotkeys
    for (const hotkey of this.hotkeys.values()) {
      if (!hotkey.enabled) continue;

      if (this.matchesCombination(event, hotkey.combination)) {
        if (hotkey.preventDefault) {
          event.preventDefault();
        }
        if (hotkey.stopPropagation) {
          event.stopPropagation();
        }

        const result = hotkey.handler(event);
        if (result === false) {
          break; // Stop processing if handler returns false
        }
      }
    }
  };

  /**
   * Starts listening for hotkeys
   */
  start(): void {
    this.scope.addEventListener('keydown', this.handleKeyDown);
  }

  /**
   * Stops listening for hotkeys
   */
  stop(): void {
    this.scope.removeEventListener('keydown', this.handleKeyDown);
  }

  /**
   * Gets all registered hotkeys
   */
  getAllHotkeys(): HotkeyRegistration[] {
    return Array.from(this.hotkeys.values());
  }

  /**
   * Gets enabled hotkeys
   */
  getEnabledHotkeys(): HotkeyRegistration[] {
    return Array.from(this.hotkeys.values()).filter((h) => h.enabled);
  }

  /**
   * Gets hotkey by ID
   */
  getHotkey(id: string): HotkeyRegistration | undefined {
    return this.hotkeys.get(id);
  }

  /**
   * Clears all hotkeys
   */
  clear(): void {
    this.hotkeys.clear();
  }
}

/**
 * Global hotkeys manager instance
 */
export const advancedHotkeysManager = new AdvancedHotkeysManager();

// Start listening
advancedHotkeysManager.start();



