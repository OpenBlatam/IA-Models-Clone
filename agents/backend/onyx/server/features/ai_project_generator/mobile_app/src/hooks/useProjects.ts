import { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import { Project, ApiError } from '../types';

export const useProjects = (params?: {
  status?: string;
  author?: string;
  limit?: number;
  offset?: number;
}) => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
    try {
      setError(null);
      setLoading(true);
      const data = await apiService.listProjects(params);
      setProjects(data);
    } catch (err) {
      setError(err as ApiError);
    } finally {
      setLoading(false);
    }
  };

  return { projects, loading, error, refetch: loadProjects };
};

