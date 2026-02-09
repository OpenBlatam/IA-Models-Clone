/**
 * Utilidades para construir contexto de tareas
 */

export interface TaskContext {
  repo_info?: {
    name: string;
    full_name: string;
    description?: string;
    language?: string;
    default_branch?: string;
    html_url?: string;
  };
  metadata?: {
    repository_id?: number;
  };
}

/**
 * Construye el contexto de una tarea a partir de su información
 */
export function buildTaskContext(task: any): TaskContext {
  if (!task.repoInfo) {
    return {};
  }

  return {
    repo_info: {
      name: task.repoInfo.name,
      full_name: task.repoInfo.full_name,
      description: task.repoInfo.description,
      language: task.repoInfo.language,
      default_branch: task.repoInfo.default_branch,
      html_url: task.repoInfo.html_url,
    },
    metadata: {
      repository_id: task.repoInfo.id,
    },
  };
}

/**
 * Determina el endpoint de API según el modelo
 */
export function getApiEndpoint(model: string, baseUrl: string): string {
  const isOpenRouter = 
    model.includes('openrouter') || 
    model.includes('gpt') || 
    model.includes('claude');
  
  return isOpenRouter
    ? `${baseUrl}/api/openrouter/stream`
    : `${baseUrl}/api/deepseek/stream`;
}

