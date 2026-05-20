/**
 * Clipboard utility functions.
 * Provides helper functions for clipboard operations.
 */

/**
 * Copies text to clipboard.
 * Falls back to legacy method if Clipboard API is not available.
 *
 * @param text - Text to copy
 * @returns Promise that resolves to true if successful
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  if (typeof navigator === 'undefined') {
    return false;
  }

  // Use modern Clipboard API if available
  if (navigator.clipboard && navigator.clipboard.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Fall through to legacy method
    }
  }

  // Fallback to legacy method
  try {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();

    const successful = document.execCommand('copy');
    document.body.removeChild(textArea);
    return successful;
  } catch {
    return false;
  }
}

/**
 * Reads text from clipboard.
 * @returns Promise that resolves to clipboard text or null
 */
export async function readFromClipboard(): Promise<string | null> {
  if (typeof navigator === 'undefined') {
    return null;
  }

  if (navigator.clipboard && navigator.clipboard.readText) {
    try {
      return await navigator.clipboard.readText();
    } catch {
      return null;
    }
  }

  return null;
}

