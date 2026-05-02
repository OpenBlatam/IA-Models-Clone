'use client';

import apiClient from '@/src/api/client';
import { useApiTask } from './useApiTask';
import type { ProductDetailsRequest, ProductDetailsResult } from '@/src/types/api';

export interface UseProductDetailsReturn {
    getDetails: (request: ProductDetailsRequest) => Promise<void>;
    result: ProductDetailsResult | null;
    isLoading: boolean;
    isPolling: boolean;
    error: string | null;
    reset: () => void;
}

export const useProductDetails = (): UseProductDetailsReturn => {
    const { execute, result, isLoading, isPolling, error, reset } = useApiTask<ProductDetailsRequest, ProductDetailsResult>(
        (req) => apiClient.getProductDetails(req),
        'Getting product details failed'
    );

    return {
        getDetails: execute,
        result,
        isLoading,
        isPolling,
        error,
        reset,
    };
};

export default useProductDetails;
