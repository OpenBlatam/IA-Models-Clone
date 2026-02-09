import { useCallback } from "react";
import { toast } from "sonner";
import { container } from "../../infrastructure/container/DIContainer";
import type { AgentEntity } from "../../domain/entities/Agent";

export function useAgentActionsContainer(onSuccess?: () => void) {
  const toggleActive = useCallback(
    async (agent: AgentEntity) => {
      try {
        await container.toggleAgentUseCase.execute(agent);
        toast.success(agent.isActive ? "Agente pausado" : "Agente activado");
        onSuccess?.();
      } catch (error) {
        const message = error instanceof Error ? error.message : "Error al actualizar agente";
        toast.error(message);
      }
    },
    [onSuccess]
  );

  return { toggleActive };
}
