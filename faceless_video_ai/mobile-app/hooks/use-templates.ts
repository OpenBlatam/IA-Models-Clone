import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { templateService } from '@/services/template-service';

export function useTemplates() {
  return useQuery({
    queryKey: ['templates'],
    queryFn: () => templateService.listTemplates(),
  });
}

export function useTemplate(name: string, enabled = true) {
  return useQuery({
    queryKey: ['template', name],
    queryFn: () => templateService.getTemplate(name),
    enabled: enabled && !!name,
  });
}

export function useGenerateFromTemplate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      templateName,
      scriptText,
      language,
    }: {
      templateName: string;
      scriptText: string;
      language?: string;
    }) => templateService.generateFromTemplate(templateName, scriptText, language),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['videos'] });
    },
  });
}

export function useCustomTemplates(userOnly = false) {
  return useQuery({
    queryKey: ['custom-templates', userOnly],
    queryFn: () => templateService.listCustomTemplates(userOnly),
  });
}

export function useCreateCustomTemplate() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      name,
      description,
      config,
      isPublic,
    }: {
      name: string;
      description: string;
      config: {
        video_config: unknown;
        audio_config: unknown;
        subtitle_config: unknown;
      };
      isPublic?: boolean;
    }) => templateService.createCustomTemplate(name, description, config, isPublic),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['custom-templates'] });
    },
  });
}


