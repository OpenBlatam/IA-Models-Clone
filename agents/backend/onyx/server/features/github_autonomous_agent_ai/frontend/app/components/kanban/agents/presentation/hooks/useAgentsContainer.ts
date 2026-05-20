import { useState, useEffect, useCallback } from "react";
import { container } from "../../infrastructure/container/DIContainer";
import { REFRESH_INTERVAL_MS } from "../../config/constants";
import type { AgentEntity } from "../../domain/entities/Agent";

export function useAgentsContainer() {
  const [agents, setAgents] = useState<AgentEntity[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchAgents = useCallback(async () => {
    try {
      setError(null);
      setIsLoading(true);
      const result = await container.getAgentsUseCase.execute();
      // Ensure result is always an array
      setAgents(Array.isArray(result) ? result : []);
    } catch (err) {
      const error = err instanceof Error ? err : new Error("Unknown error");
      setError(error);
      console.error("Error fetching agents", error);
      setAgents([]); // Set empty array on error
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAgents();
    const interval = setInterval(fetchAgents, REFRESH_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchAgents]);

  return {
    agents,
    isLoading,
    error,
    refetch: fetchAgents,
  };
}
