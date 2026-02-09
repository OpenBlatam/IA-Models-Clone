/**
 * Hook para monitorear el estado del agente.
 */

import { useState, useEffect, useCallback } from 'react';
import { getAPIClient } from '../lib/api-client';
import { AgentStatus } from '../lib/api-client';

interface UseAgentStatusOptions {
  enabled?: boolean;
  pollInterval?: number;
}

/**
 * Hook para obtener y monitorear el estado del agente.
 */
export function useAgentStatus(options: UseAgentStatusOptions = {}) {
  const {
    enabled = true,
    pollInterval = 5000
  } = options;

  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    if (!enabled) return;

    try {
      const client = getAPIClient();
      const agentStatus = await client.getAgentStatus();
      setStatus(agentStatus);
      setError(null);
    } catch (err: any) {
      setError(err.message || 'Error obteniendo estado del agente');
      console.error('Error obteniendo estado del agente:', err);
    } finally {
      setIsLoading(false);
    }
  }, [enabled]);

  const startAgent = useCallback(async () => {
    try {
      const client = getAPIClient();
      await client.startAgent();
      await fetchStatus();
    } catch (err: any) {
      throw new Error(err.message || 'Error iniciando agente');
    }
  }, [fetchStatus]);

  const stopAgent = useCallback(async () => {
    try {
      const client = getAPIClient();
      await client.stopAgent();
      await fetchStatus();
    } catch (err: any) {
      throw new Error(err.message || 'Error deteniendo agente');
    }
  }, [fetchStatus]);

  const pauseAgent = useCallback(async () => {
    try {
      const client = getAPIClient();
      await client.pauseAgent();
      await fetchStatus();
    } catch (err: any) {
      throw new Error(err.message || 'Error pausando agente');
    }
  }, [fetchStatus]);

  const resumeAgent = useCallback(async () => {
    try {
      const client = getAPIClient();
      await client.resumeAgent();
      await fetchStatus();
    } catch (err: any) {
      throw new Error(err.message || 'Error reanudando agente');
    }
  }, [fetchStatus]);

  // Polling automático
  useEffect(() => {
    if (!enabled) return;

    fetchStatus();

    const interval = setInterval(fetchStatus, pollInterval);
    return () => clearInterval(interval);
  }, [enabled, pollInterval, fetchStatus]);

  return {
    status,
    isLoading,
    error,
    fetchStatus,
    startAgent,
    stopAgent,
    pauseAgent,
    resumeAgent
  };
}



