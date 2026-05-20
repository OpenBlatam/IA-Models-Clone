/**
 * Clipboard utilities
 */

// Copy text to clipboard
export async function copyToClipboard(text: string): Promise<boolean> {
  if (typeof window === 'undefined' || !navigator.clipboard) {
    // Fallback for older browsers
    return fallbackCopyToClipboard(text);
  }

  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (error) {
    console.error('Failed to copy to clipboard:', error);
    return fallbackCopyToClipboard(text);
  }
}

// Fallback copy method
function fallbackCopyToClipboard(text: string): boolean {
  if (typeof document === 'undefined') {
    return false;
  }

  const textArea = document.createElement('textarea');
  textArea.value = text;
  textArea.style.position = 'fixed';
  textArea.style.left = '-999999px';
  textArea.style.top = '-999999px';
  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();

  try {
    const successful = document.execCommand('copy');
    document.body.removeChild(textArea);
    return successful;
  } catch (error) {
    document.body.removeChild(textArea);
    console.error('Fallback copy failed:', error);
    return false;
  }
}

// Read from clipboard
export async function readFromClipboard(): Promise<string> {
  if (typeof window === 'undefined' || !navigator.clipboard) {
    throw new Error('Clipboard API not available');
  }

  try {
    return await navigator.clipboard.readText();
  } catch (error) {
    console.error('Failed to read from clipboard:', error);
    throw error;
  }
}

// Copy image to clipboard
export async function copyImageToClipboard(blob: Blob): Promise<boolean> {
  if (typeof window === 'undefined' || !navigator.clipboard) {
    return false;
  }

  try {
    const item = new ClipboardItem({ 'image/png': blob });
    await navigator.clipboard.write([item]);
    return true;
  } catch (error) {
    console.error('Failed to copy image to clipboard:', error);
    return false;
  }
}



