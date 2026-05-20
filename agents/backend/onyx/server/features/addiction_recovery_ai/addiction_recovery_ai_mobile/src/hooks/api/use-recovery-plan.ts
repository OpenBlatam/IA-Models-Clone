import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { recoveryPlanApi } from '@/services/api';

export function useRecoveryPlan(userId: string | null) {
  return useQuery({
    queryKey: ['recovery-plan', userId],
    queryFn: () => recoveryPlanApi.getPlan(userId!),
    enabled: !!userId,
  });
}

export function useCreatePlan() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: recoveryPlanApi.createPlan,
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['recovery-plan', variables.user_id] });
    },
  });
}

export function useStrategies(addictionType: string | null) {
  return useQuery({
    queryKey: ['strategies', addictionType],
    queryFn: () => recoveryPlanApi.getStrategies(addictionType!),
    enabled: !!addictionType,
  });
}

