import { showToast } from '@/lib/toast';

export interface ApiError {
  message: string;
  code?: string;
  status?: number;
}

export function handleApiError(error: any): ApiError {
  if (error.response) {
    // Server responded with error
    return {
      message: error.response.data?.message || error.response.data?.error || 'Error del servidor',
      code: error.response.data?.code,
      status: error.response.status,
    };
  } else if (error.request) {
    // Request made but no response
    return {
      message: 'No se pudo conectar con el servidor',
      code: 'NETWORK_ERROR',
    };
  } else {
    // Something else happened
    return {
      message: error.message || 'Error desconocido',
      code: 'UNKNOWN_ERROR',
    };
  }
}

export async function withErrorHandling<T>(
  fn: () => Promise<T>,
  errorMessage?: string
): Promise<T | null> {
  try {
    return await fn();
  } catch (error: any) {
    const apiError = handleApiError(error);
    showToast(errorMessage || apiError.message, 'error');
    console.error('API Error:', apiError);
    return null;
  }
}

export function createQueryString(params: Record<string, any>): string {
  const searchParams = new URLSearchParams();
  
  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== '') {
      if (Array.isArray(value)) {
        value.forEach((item) => searchParams.append(key, String(item)));
      } else {
        searchParams.append(key, String(value));
      }
    }
  });
  
  return searchParams.toString();
}

export function parseQueryString(queryString: string): Record<string, string | string[]> {
  const params = new URLSearchParams(queryString);
  const result: Record<string, string | string[]> = {};
  
  params.forEach((value, key) => {
    if (result[key]) {
      const existing = result[key];
      result[key] = Array.isArray(existing) 
        ? [...existing, value]
        : [existing as string, value];
    } else {
      result[key] = value;
    }
  });
  
  return result;
}

