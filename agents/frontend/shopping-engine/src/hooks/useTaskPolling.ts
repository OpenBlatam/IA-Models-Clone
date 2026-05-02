'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import apiClient from '@/src/api/client';
import type { TaskStatusResponse } from '@/src/types/api';

interface UseTaskPollingOptions<T> {
    pollingInterval?: number;
    maxAttempts?: number;
    onComplete?: (result: T) => void;
    onError?: (error: string) => void;
}

interface UseTaskPollingReturn<T> {
    status: TaskStatusResponse | null;
    result: T | null;
    isPolling: boolean;
    error: string | null;
    startPolling: (taskId: string) => void;
    stopPolling: () => void;
}

export const useTaskPolling = <T>(
    options: UseTaskPollingOptions<T> = {}
): UseTaskPollingReturn<T> => {
    const {
        pollingInterval = 2000,
        maxAttempts = 60,
        onComplete,
        onError,
    } = options;

    const [taskId, setTaskId] = useState<string | null>(null);
    const [status, setStatus] = useState<TaskStatusResponse | null>(null);
    const [result, setResult] = useState<T | null>(null);
    const [isPolling, setIsPolling] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [attempts, setAttempts] = useState(0);

    const onCompleteRef = useRef(onComplete);
    const onErrorRef = useRef(onError);

    useEffect(() => {
        onCompleteRef.current = onComplete;
        onErrorRef.current = onError;
    }, [onComplete, onError]);

    const stopPolling = useCallback(() => {
        setIsPolling(false);
        setTaskId(null);
        setAttempts(0);
    }, []);

    const startPolling = useCallback((newTaskId: string) => {
        setTaskId(newTaskId);
        setIsPolling(true);
        setError(null);
        setResult(null);
        setStatus(null);
        setAttempts(0);
    }, []);

    useEffect(() => {
        let isCancelled = false;

        if (!isPolling || !taskId) {
            return;
        }

        const pollStatus = async () => {
            try {
                const statusResponse = await apiClient.getTaskStatus(taskId);
                if (isCancelled) return;
                
                setStatus(statusResponse);

                if (statusResponse.status === 'completed') {
                    const taskResult = await apiClient.getTaskResult<T>(taskId);
                    if (isCancelled) return;
                    
                    setResult(taskResult);
                    stopPolling();
                    onCompleteRef.current?.(taskResult);
                    return;
                }

                if (statusResponse.status === 'failed') {
                    const errorMessage = statusResponse.error || 'Task failed';
                    setError(errorMessage);
                    stopPolling();
                    onErrorRef.current?.(errorMessage);
                    return;
                }

                setAttempts((prev) => prev + 1);
            } catch (err) {
                if (isCancelled) return;
                const errorMessage = err instanceof Error ? err.message : 'Polling failed';
                setError(errorMessage);
                stopPolling();
                onErrorRef.current?.(errorMessage);
            }
        };

        if (attempts >= maxAttempts) {
            const timeoutError = 'Task polling timeout';
            setError(timeoutError);
            stopPolling();
            onErrorRef.current?.(timeoutError);
            return;
        }

        const timeoutId = setTimeout(pollStatus, attempts === 0 ? 0 : pollingInterval);

        return () => {
            isCancelled = true;
            clearTimeout(timeoutId);
        };
    }, [isPolling, taskId, attempts, pollingInterval, maxAttempts, stopPolling]);

    return {
        status,
        result,
        isPolling,
        error,
        startPolling,
        stopPolling,
    };
};

export default useTaskPolling;
