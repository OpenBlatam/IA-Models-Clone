'use client';

import apiClient from '@/src/api/client';
import { useApiTask } from './useApiTask';
import type { FindPurchaseOptionsRequest, PurchaseOptionsResult } from '@/src/types/api';

export interface UsePurchaseOptionsReturn {
    findOptions: (request: FindPurchaseOptionsRequest) => Promise<void>;
    result: PurchaseOptionsResult | null;
    isLoading: boolean;
    isPolling: boolean;
    error: string | null;
    reset: () => void;
}

export const usePurchaseOptions = (): UsePurchaseOptionsReturn => {
    const { execute, result, isLoading, isPolling, error, reset } = useApiTask<FindPurchaseOptionsRequest, PurchaseOptionsResult>(
        (req) => apiClient.findPurchaseOptions(req),
        'Finding purchase options failed'
    );

    return {
        findOptions: execute,
        result,
        isLoading,
        isPolling,
        error,
        reset,
    };
};

export default usePurchaseOptions;
