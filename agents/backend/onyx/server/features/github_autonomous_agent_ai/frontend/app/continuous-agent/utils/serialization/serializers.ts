/**
 * Serialization utilities
 * 
 * Provides functions to serialize and deserialize data
 */

/**
 * Serializes data to JSON string with error handling
 */
export function serialize<T>(data: T): string {
  try {
    return JSON.stringify(data, null, 2);
  } catch (error) {
    throw new Error(`Failed to serialize data: ${error instanceof Error ? error.message : String(error)}`);
  }
}

/**
 * Deserializes JSON string to data with error handling
 */
export function deserialize<T>(json: string): T {
  try {
    return JSON.parse(json) as T;
  } catch (error) {
    throw new Error(`Failed to deserialize JSON: ${error instanceof Error ? error.message : String(error)}`);
  }
}

/**
 * Safely serializes data, returns null on error
 */
export function safeSerialize<T>(data: T): string | null {
  try {
    return serialize(data);
  } catch {
    return null;
  }
}

/**
 * Safely deserializes JSON, returns null on error
 */
export function safeDeserialize<T>(json: string): T | null {
  try {
    return deserialize<T>(json);
  } catch {
    return null;
  }
}

/**
 * Serializes with custom replacer
 */
export function serializeWithReplacer<T>(
  data: T,
  replacer?: (key: string, value: unknown) => unknown
): string {
  try {
    return JSON.stringify(data, replacer, 2);
  } catch (error) {
    throw new Error(`Failed to serialize data: ${error instanceof Error ? error.message : String(error)}`);
  }
}

/**
 * Deserializes with custom reviver
 */
export function deserializeWithReviver<T>(
  json: string,
  reviver?: (key: string, value: unknown) => unknown
): T {
  try {
    return JSON.parse(json, reviver) as T;
  } catch (error) {
    throw new Error(`Failed to deserialize JSON: ${error instanceof Error ? error.message : String(error)}`);
  }
}

/**
 * Serializes to base64
 */
export function serializeToBase64<T>(data: T): string {
  const json = serialize(data);
  return btoa(encodeURIComponent(json));
}

/**
 * Deserializes from base64
 */
export function deserializeFromBase64<T>(base64: string): T {
  const json = decodeURIComponent(atob(base64));
  return deserialize<T>(json);
}

/**
 * Creates a deep clone using serialization
 */
export function deepClone<T>(obj: T): T {
  return deserialize<T>(serialize(obj));
}

/**
 * Serializes with circular reference handling
 */
export function serializeCircular<T>(data: T): string {
  const seen = new WeakSet();
  
  return JSON.stringify(data, (key, value) => {
    if (typeof value === "object" && value !== null) {
      if (seen.has(value)) {
        return "[Circular]";
      }
      seen.add(value);
    }
    return value;
  }, 2);
}




