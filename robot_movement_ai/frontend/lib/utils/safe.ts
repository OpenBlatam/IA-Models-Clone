/**
 * Safe operation utilities (prevent errors)
 */

// Safe JSON parse
export function safeJsonParse<T>(json: string, defaultValue: T): T {
  try {
    return JSON.parse(json) as T;
  } catch {
    return defaultValue;
  }
}

// Safe JSON stringify
export function safeJsonStringify(value: any, defaultValue: string = ''): string {
  try {
    return JSON.stringify(value);
  } catch {
    return defaultValue;
  }
}

// Safe number parse
export function safeNumberParse(value: any, defaultValue: number = 0): number {
  const num = Number(value);
  return isNaN(num) ? defaultValue : num;
}

// Safe integer parse
export function safeIntParse(value: any, defaultValue: number = 0): number {
  const num = parseInt(String(value), 10);
  return isNaN(num) ? defaultValue : num;
}

// Safe float parse
export function safeFloatParse(value: any, defaultValue: number = 0): number {
  const num = parseFloat(String(value));
  return isNaN(num) ? defaultValue : num;
}

// Safe division
export function safeDivide(numerator: number, denominator: number, defaultValue: number = 0): number {
  if (denominator === 0) return defaultValue;
  return numerator / denominator;
}

// Safe access to nested property
export function safeGet<T>(obj: any, path: string, defaultValue: T): T {
  try {
    const keys = path.split('.');
    let current = obj;

    for (const key of keys) {
      if (current === null || current === undefined) {
        return defaultValue;
      }
      current = current[key];
    }

    return current !== undefined ? current : defaultValue;
  } catch {
    return defaultValue;
  }
}

// Safe function call
export function safeCall<T>(
  fn: () => T,
  defaultValue: T,
  onError?: (error: unknown) => void
): T {
  try {
    return fn();
  } catch (error) {
    if (onError) {
      onError(error);
    }
    return defaultValue;
  }
}

// Safe async function call
export async function safeAsyncCall<T>(
  fn: () => Promise<T>,
  defaultValue: T,
  onError?: (error: unknown) => void
): Promise<T> {
  try {
    return await fn();
  } catch (error) {
    if (onError) {
      onError(error);
    }
    return defaultValue;
  }
}



