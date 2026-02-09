import type { AgentError } from "../utils/errors/agent-errors";
import { toAgentError, AgentServerError } from "../utils/errors/agent-errors";
import { getApiErrorMessage } from "../utils/apiError";
import { validateWithZod, safeValidateWithZod, type ZodSchema } from "../utils/validation/zod-validator";

const DEFAULT_HEADERS = {
  "Content-Type": "application/json",
} as const;

/**
 * Type guard to check if error response has detail field
 */
const isErrorResponse = (value: unknown): value is { detail: string } => {
  return (
    typeof value === "object" &&
    value !== null &&
    "detail" in value &&
    typeof (value as { detail: unknown }).detail === "string"
  );
};

/**
 * Handles HTTP response and extracts error messages if needed
 */
const handleResponse = async <T>(response: Response, defaultError: string): Promise<T> => {
  // Check content type before parsing
  const contentType = response.headers.get("content-type") || "";
  const isJson = contentType.includes("application/json");
  
  if (response.ok) {
    if (!isJson) {
      // Only log in development
      if (process.env.NODE_ENV === "development") {
        const text = await response.text();
        console.warn("API returned non-JSON response:", {
          contentType,
          preview: text.substring(0, 200),
        });
      }
      throw new Error(`API returned non-JSON response. Content-Type: ${contentType}`);
    }
    try {
      return await response.json();
    } catch (error) {
      if (error instanceof SyntaxError && error.message.includes("JSON")) {
        // Only log in development
        if (process.env.NODE_ENV === "development") {
          const text = await response.text();
          console.warn("Failed to parse JSON response:", text.substring(0, 200));
        }
        throw new Error("Failed to parse JSON response. The API may have returned HTML or invalid JSON.");
      }
      throw error;
    }
  }

  // Handle error responses
  try {
    if (isJson) {
      const error = await response.json();
      const message = isErrorResponse(error) ? error.detail : defaultError;
      throw new Error(message);
    } else {
      // If error response is HTML, extract status info
      // Only log in development to avoid console noise
      if (process.env.NODE_ENV === "development") {
        const text = await response.text();
        console.warn("API returned HTML instead of JSON:", {
          status: response.status,
          statusText: response.statusText,
          contentType,
          preview: text.substring(0, 200),
        });
      }
      
      // Provide user-friendly error message
      const statusMessage = response.status === 404
        ? "The API endpoint was not found. Please check if the backend server is running."
        : response.status === 500
        ? "The backend server encountered an error."
        : `The API returned an error (${response.status} ${response.statusText}).`;
      
      throw new Error(statusMessage);
    }
  } catch (err) {
    if (err instanceof Error) {
      throw err;
    }
    throw new Error(defaultError);
  }
};

/**
 * Options for HTTP client requests
 */
type HttpClientOptions = {
  readonly method?: "GET" | "POST" | "PATCH" | "DELETE" | "PUT";
  readonly headers?: Record<string, string>;
  readonly body?: unknown;
  readonly validateResponse?: ZodSchema<unknown>;
  readonly validateRequest?: ZodSchema<unknown>;
  readonly defaultError?: string;
};

/**
 * Generic HTTP client for agent service requests
 * 
 * Features:
 * - Automatic error handling
 * - Request/response validation with Zod
 * - Type-safe operations
 * - Consistent error messages
 */
export const httpClient = async <T>(
  url: string,
  options: HttpClientOptions = {}
): Promise<T> => {
  const {
    method = "GET",
    headers = {},
    body,
    validateResponse,
    validateRequest,
    defaultError = "Error en la solicitud",
  } = options;

  try {
    // Validate request body if schema provided
    let validatedBody = body;
    if (body && validateRequest) {
      validatedBody = validateWithZod(validateRequest, body, "Invalid request data");
    }

    // Execute request
    const response = await fetch(url, {
      method,
      headers: { ...DEFAULT_HEADERS, ...headers },
      body: validatedBody ? JSON.stringify(validatedBody) : undefined,
    });

    const data = await handleResponse<T>(response, defaultError);

    // Validate response if schema provided
    if (validateResponse) {
      if (Array.isArray(data)) {
        const validationResult = safeValidateWithZod(validateResponse.array(), data);
        if (!validationResult.success) {
          throw new AgentServerError("Los datos recibidos no son válidos", {
            cause: validationResult.error,
          });
        }
        return validationResult.data as T;
      }
      return validateWithZod(validateResponse, data, "Invalid response data");
    }

    return data;
  } catch (error) {
    // Handle network errors specifically before converting to AgentError
    if (
      error instanceof TypeError ||
      (error instanceof Error && (
        error.message.includes("fetch") ||
        error.message.includes("Failed to fetch") ||
        error.message.includes("NetworkError") ||
        error.message.includes("network") ||
        error.message.includes("ECONNREFUSED") ||
        error.message.includes("ENOTFOUND")
      ))
    ) {
      // Provide a more helpful error message for network issues
      const networkError = new Error(
        "Cannot connect to the server. Please check if the backend is running and accessible."
      );
      throw toAgentError(networkError);
    }
    throw toAgentError(error);
  }
};

/**
 * GET request helper
 */
export const httpGet = async <T>(
  url: string,
  schema?: ZodSchema<T>,
  defaultError = "Error al cargar los datos"
): Promise<T> => {
  return httpClient<T>(url, {
    method: "GET",
    validateResponse: schema,
    defaultError,
  });
};

/**
 * POST request helper
 */
export const httpPost = async <TRequest, TResponse>(
  url: string,
  body: TRequest,
  requestSchema?: ZodSchema<TRequest>,
  responseSchema?: ZodSchema<TResponse>,
  defaultError = "Error al crear el recurso"
): Promise<TResponse> => {
  return httpClient<TResponse>(url, {
    method: "POST",
    body,
    validateRequest: requestSchema,
    validateResponse: responseSchema,
    defaultError,
  });
};

/**
 * PATCH request helper
 */
export const httpPatch = async <TRequest, TResponse>(
  url: string,
  body: TRequest,
  requestSchema?: ZodSchema<TRequest>,
  responseSchema?: ZodSchema<TResponse>,
  defaultError = "Error al actualizar el recurso"
): Promise<TResponse> => {
  return httpClient<TResponse>(url, {
    method: "PATCH",
    body,
    validateRequest: requestSchema,
    validateResponse: responseSchema,
    defaultError,
  });
};

/**
 * DELETE request helper
 */
export const httpDelete = async (
  url: string,
  defaultError = "Error al eliminar el recurso"
): Promise<void> => {
  return httpClient<void>(url, {
    method: "DELETE",
    defaultError,
  });
};



