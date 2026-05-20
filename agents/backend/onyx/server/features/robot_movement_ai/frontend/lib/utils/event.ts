/**
 * Event utilities
 */

// Create custom event
export function createCustomEvent<T = any>(name: string, detail?: T): CustomEvent<T> {
  return new CustomEvent(name, { detail });
}

// Dispatch custom event
export function dispatchCustomEvent<T = any>(name: string, detail?: T): boolean {
  if (typeof window === 'undefined') {
    return false;
  }
  const event = createCustomEvent(name, detail);
  return window.dispatchEvent(event);
}

// Listen to custom event
export function listenToCustomEvent<T = any>(
  name: string,
  handler: (event: CustomEvent<T>) => void
): () => void {
  if (typeof window === 'undefined') {
    return () => {};
  }

  const eventHandler = (e: Event) => {
    handler(e as CustomEvent<T>);
  };

  window.addEventListener(name, eventHandler);
  return () => window.removeEventListener(name, eventHandler);
}

// Prevent default and stop propagation
export function stopEvent(event: React.SyntheticEvent | Event) {
  event.preventDefault();
  event.stopPropagation();
}

// Get event target value
export function getEventValue(event: React.ChangeEvent<HTMLInputElement>): string {
  return event.target.value;
}

// Get event target checked
export function getEventChecked(event: React.ChangeEvent<HTMLInputElement>): boolean {
  return event.target.checked;
}



