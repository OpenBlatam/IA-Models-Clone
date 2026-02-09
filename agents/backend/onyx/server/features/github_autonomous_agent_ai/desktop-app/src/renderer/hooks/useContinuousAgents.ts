import { useState, useEffect, useCallback } from 'react';
import { agentService } from '../services/agentService';
import type {
  ContinuousAgent,
  CreateAgentRequest,
  UpdateAgentRequest,
} from '../types/agent';
import { toast } from 'sonner';

interface UseContinuousAgentsOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export function useContinuousAgents(options: UseContinuousAgentsOptions = {}) {
  const { autoRefresh = true, refreshInterval = 5000 } = options;

  const [agents, setAgents] = useState<ContinuousAgent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAgents = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await agentService.fetchAgents();
      setAgents(data);
    } catch (err: any) {
      const errorMessage =
        err.response?.data?.detail || err.message || 'Error al cargar agentes';
      setError(errorMessage);
      console.error('Error fetching agents:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAgents();

    if (autoRefresh) {
      const interval = setInterval(fetchAgents, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [fetchAgents, autoRefresh, refreshInterval]);

  const createAgent = useCallback(
    async (request: CreateAgentRequest): Promise<ContinuousAgent> => {
      try {
        if (!request.name?.trim() || !request.description?.trim()) {
          throw new Error('Name and description are required');
        }
        const newAgent = await agentService.createAgent(request);
        setAgents((prev) => [...prev, newAgent]);
        toast.success('Agente creado exitosamente');
        return newAgent;
      } catch (err: any) {
        const errorMessage =
          err.response?.data?.detail || err.message || 'Error al crear agente';
        toast.error(errorMessage);
        throw err;
      }
    },
    []
  );

  const updateAgent = useCallback(
    async (
      agentId: string,
      updates: UpdateAgentRequest
    ): Promise<ContinuousAgent> => {
      try {
        const updatedAgent = await agentService.updateAgent(agentId, updates);
        setAgents((prev) =>
          prev.map((agent) => (agent.id === agentId ? updatedAgent : agent))
        );
        toast.success('Agente actualizado exitosamente');
        return updatedAgent;
      } catch (err: any) {
        const errorMessage =
          err.response?.data?.detail || err.message || 'Error al actualizar agente';
        toast.error(errorMessage);
        throw err;
      }
    },
    []
  );

  const deleteAgent = useCallback(async (agentId: string): Promise<void> => {
    try {
      await agentService.deleteAgent(agentId);
      setAgents((prev) => prev.filter((agent) => agent.id !== agentId));
      toast.success('Agente eliminado exitosamente');
    } catch (err: any) {
      const errorMessage =
        err.response?.data?.detail || err.message || 'Error al eliminar agente';
      toast.error(errorMessage);
      throw err;
    }
  }, []);

  const toggleAgent = useCallback(
    async (agentId: string): Promise<void> => {
      try {
        const updatedAgent = await agentService.toggleAgent(agentId);
        setAgents((prev) =>
          prev.map((agent) => (agent.id === agentId ? updatedAgent : agent))
        );
        toast.success(
          `Agente ${updatedAgent.isActive ? 'activado' : 'desactivado'}`
        );
      } catch (err: any) {
        const errorMessage =
          err.response?.data?.detail || err.message || 'Error al cambiar estado';
        toast.error(errorMessage);
        throw err;
      }
    },
    []
  );

  return {
    agents,
    loading,
    error,
    fetchAgents,
    createAgent,
    updateAgent,
    deleteAgent,
    toggleAgent,
  };
}


