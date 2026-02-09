/**
 * Input sanitization utilities for security
 * 
 * Provides functions to sanitize user input and prevent XSS attacks
 */

/**
 * Sanitizes a string by removing potentially dangerous characters
 * @param input - String to sanitize
 * @returns Sanitized string
 */
export function sanitizeString(input: string): string {
  if (typeof input !== "string") {
    return "";
  }

  return input
    .trim()
    .replace(/[<>]/g, "") // Remove angle brackets
    .replace(/javascript:/gi, "") // Remove javascript: protocol
    .replace(/on\w+=/gi, ""); // Remove event handlers
}

/**
 * Sanitizes an object by sanitizing all string values
 * @param obj - Object to sanitize
 * @returns Sanitized object
 */
export function sanitizeObject<T extends Record<string, unknown>>(
  obj: T
): T {
  const sanitized = { ...obj };

  for (const key in sanitized) {
    if (typeof sanitized[key] === "string") {
      sanitized[key] = sanitizeString(sanitized[key] as string) as T[typeof key];
    } else if (
      typeof sanitized[key] === "object" &&
      sanitized[key] !== null &&
      !Array.isArray(sanitized[key])
    ) {
      sanitized[key] = sanitizeObject(
        sanitized[key] as Record<string, unknown>
      ) as T[typeof key];
    }
  }

  return sanitized;
}

/**
 * Escapes HTML special characters to prevent XSS
 * @param text - Text to escape
 * @returns Escaped text
 */
export function escapeHtml(text: string): string {
  if (typeof text !== "string") {
    return "";
  }

  const map: Record<string, string> = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  };

  return text.replace(/[&<>"']/g, (char) => map[char] || char);
}

/**
 * Validates and sanitizes JSON string
 * @param jsonString - JSON string to validate and sanitize
 * @returns Sanitized parsed object
 * @throws Error if JSON is invalid
 */
export function sanitizeJSON(jsonString: string): Record<string, unknown> {
  if (typeof jsonString !== "string") {
    throw new Error("JSON string must be a string");
  }

  try {
    const parsed = JSON.parse(jsonString);

    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      throw new Error("JSON must be an object");
    }

    return sanitizeObject(parsed);
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error("Invalid JSON");
  }
}

/**
 * Sanitizes agent name
 * @param name - Agent name to sanitize
 * @returns Sanitized name
 */
export function sanitizeAgentName(name: string): string {
  return sanitizeString(name).slice(0, 100); // Enforce max length
}

/**
 * Sanitizes agent description
 * @param description - Agent description to sanitize
 * @returns Sanitized description
 */
export function sanitizeAgentDescription(description: string): string {
  return sanitizeString(description).slice(0, 500); // Enforce max length
}




