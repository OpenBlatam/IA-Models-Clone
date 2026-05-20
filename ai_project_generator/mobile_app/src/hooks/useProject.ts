import { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import { Project, ApiError } from '../types';

export const useProject = (projectId: string) => {
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => {
    loadProject();
  }, [projectId]);

  const loadProject = async () => {
    try {
      setError(null);
      setLoading(true);
      const data = await apiService.getProject(projectId);
      setProject(data);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  };

  return { project, loading, error, refetch: loadProject };
};

