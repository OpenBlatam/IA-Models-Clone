'use client';

import { useState, useCallback } from 'react';
import { useTaskPolling } from './useTaskPolling';
import type { DirectResultResponse, TaskSubmittedResponse } from '@/src/types/api';

export interface UseApiTaskReturn<TRequest, TResult> {
    execute: (request: TRequest) => Promise<void>;
    result: TResult | null;
    isLoading: boolean;
    isPolling: boolean;
    error: string | null;
    reset: () => void;
}

export function useApiTask<TRequest, TResult>(
    apiCall: (request: TRequest) => Promise<any>,
    defaultErrorMessage: string
): UseApiTaskReturn<TRequest, TResult> {
    const [result, setResult] = useState<TResult | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const { isPolling, startPolling, stopPolling } = useTaskPolling<TResult>({
        onComplete: (taskResult) => {
            setResult(taskResult);
            setIsLoading(false);
        },
        onError: (err) => {
            setError(err);
            setIsLoading(false);
        },
    });

    const execute = useCallback(async (request: TRequest) => {
        setIsLoading(true);
        setError(null);
        setResult(null);

        try {
            const response = await apiCall(request);

            if (response.status === 'completed') {
                const directResponse = response as DirectResultResponse<TResult>;
                setResult(directResponse.result);
                setIsLoading(false);
                return;
            }

            if (response.status === 'submitted') {
                const taskResponse = response as TaskSubmittedResponse;
                startPolling(taskResponse.task_id);
                return;
            }

            throw new Error('Unexpected response status');
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : defaultErrorMessage;
            setError(errorMessage);
            setIsLoading(false);
        }
    }, [apiCall, defaultErrorMessage, startPolling]);

    const reset = useCallback(() => {
        setResult(null);
        setError(null);
        setIsLoading(false);
        stopPolling();
    }, [stopPolling]);

    return {
        execute,
        result,
        isLoading,
        isPolling,
        error,
        reset,
    };
}
