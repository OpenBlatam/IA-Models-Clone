'use client';

import apiClient from '@/src/api/client';
import { useApiTask } from './useApiTask';
import type { AnalyzeProductRequest, ProductAnalysisResult } from '@/src/types/api';

export interface UseAnalyzeProductReturn {
    analyze: (request: AnalyzeProductRequest) => Promise<void>;
    result: ProductAnalysisResult | null;
    isLoading: boolean;
    isPolling: boolean;
    error: string | null;
    reset: () => void;
}

export const useAnalyzeProduct = (): UseAnalyzeProductReturn => {
    const { execute, result, isLoading, isPolling, error, reset } = useApiTask<AnalyzeProductRequest, ProductAnalysisResult>(
        (req) => apiClient.analyzeProduct(req),
        'Analysis failed'
    );

    return {
        analyze: execute,
        result,
        isLoading,
        isPolling,
        error,
        reset,
    };
};

export default useAnalyzeProduct;
