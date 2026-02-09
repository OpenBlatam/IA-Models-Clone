import React, { useState, useEffect, useMemo } from 'react';
import { useDebounce } from 'use-debounce';
import { githubAPI, GitHubRepository } from '../lib/github-api';
import { Input } from './ui/Input';
import { Button } from './ui/Button';
import { cn } from '../utils/cn';

interface RepositorySelectorProps {
  onSelect: (repository: GitHubRepository) => void;
  selectedRepository: GitHubRepository | null;
  authenticated: boolean;
}

export const RepositorySelector: React.FC<RepositorySelectorProps> = ({
  onSelect,
  selectedRepository,
  authenticated,
}) => {
  const [repositories, setRepositories] = useState<GitHubRepository[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedSearchQuery] = useDebounce(searchQuery, 300);
  const [filterType, setFilterType] = useState<'all' | 'public' | 'private'>('all');
  const [sortBy, setSortBy] = useState<'name' | 'updated'>('updated');

  useEffect(() => {
    if (authenticated) {
      loadRepositories();
    }
  }, [authenticated]);

  const loadRepositories = async () => {
    try {
      setLoading(true);
      setError(null);
      const repos = await githubAPI.getRepositories();
      setRepositories(repos);
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Error al cargar repositorios';
      setError(errorMessage);
      console.error('Error loading repositories:', err);
    } finally {
      setLoading(false);
    }
  };

  const filteredAndSortedRepos = useMemo(() => {
    let filtered = repositories;

    // Filter by type
    if (filterType !== 'all') {
      filtered = filtered.filter(
        (repo) => (filterType === 'public') !== repo.private
      );
    }

    // Filter by search query
    if (debouncedSearchQuery) {
      const query = debouncedSearchQuery.toLowerCase();
      filtered = filtered.filter(
        (repo) =>
          repo.name.toLowerCase().includes(query) ||
          repo.full_name.toLowerCase().includes(query) ||
          (repo.description?.toLowerCase().includes(query) ?? false)
      );
    }

    // Sort
    filtered = [...filtered].sort((a, b) => {
      if (sortBy === 'name') {
        return a.name.localeCompare(b.name);
      } else {
        return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
      }
    });

    return filtered;
  }, [repositories, debouncedSearchQuery, filterType, sortBy]);

  if (!authenticated) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <p className="text-sm text-gray-500 text-center py-4">
          Autentica con GitHub para ver tus repositorios
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h3 className="text-black text-xl font-normal mb-1">Seleccionar Repositorio</h3>
          <p className="text-sm text-gray-500">{`${filteredAndSortedRepos.length} repositorio(s) encontrado(s)`}</p>
        </div>
        <div className="ml-4">
          <Button size="sm" variant="ghost" onClick={loadRepositories}>
            Actualizar
          </Button>
        </div>
      </div>
      <div>
        <div className="space-y-4">
          {/* Search */}
          <Input
            placeholder="Buscar repositorios..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            fullWidth
          />

          {/* Filters */}
          <div className="flex gap-2">
            <select
              value={filterType}
              onChange={(e) =>
                setFilterType(e.target.value as 'all' | 'public' | 'private')
              }
              className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">Todos</option>
              <option value="public">Públicos</option>
              <option value="private">Privados</option>
            </select>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'name' | 'updated')}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="updated">Más recientes</option>
              <option value="name">Por nombre</option>
            </select>
          </div>

          {/* Repository List */}
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-black"></div>
            </div>
          ) : error ? (
            <div className="text-center py-8 text-red-600">
              <p>{error}</p>
              <Button
                variant="outline"
                size="sm"
                onClick={loadRepositories}
                className="mt-4"
              >
                Reintentar
              </Button>
            </div>
          ) : filteredAndSortedRepos.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <p>No se encontraron repositorios</p>
            </div>
          ) : (
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {filteredAndSortedRepos.map((repo) => (
                <button
                  key={repo.id}
                  onClick={() => onSelect(repo)}
                  className={cn(
                    'w-full text-left p-3 rounded-lg border transition-colors',
                    selectedRepository?.id === repo.id
                      ? 'border-black bg-gray-50'
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  )}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="font-medium text-black">{repo.name}</div>
                      <div className="text-sm text-gray-500 mt-1">
                        {repo.full_name}
                      </div>
                      {repo.description && (
                        <div className="text-sm text-gray-500 mt-1 line-clamp-2">
                          {repo.description}
                        </div>
                      )}
                      <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                        {repo.language && (
                          <span>📄 {repo.language}</span>
                        )}
                        <span>⭐ {repo.stargazers_count}</span>
                        <span>🍴 {repo.forks_count}</span>
                        <span>
                          {repo.private ? '🔒 Privado' : '🌐 Público'}
                        </span>
                      </div>
                    </div>
                    {selectedRepository?.id === repo.id && (
                      <div className="ml-4">
                        <svg
                          className="w-5 h-5 text-black"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                      </div>
                    )}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};


