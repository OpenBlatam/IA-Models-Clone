'use client';

import { useState, useEffect, useMemo, useCallback } from 'react';
import { useDebounce } from 'use-debounce';
import { format } from 'date-fns';
import { es } from 'date-fns/locale';
import { toast } from 'sonner';
import { githubAPI, GitHubRepository } from '../lib/github-api';
import { cn } from '../utils/cn';

interface RepositorySelectorProps {
  onSelect: (repository: GitHubRepository) => void;
  selectedRepository: GitHubRepository | null;
  authenticated: boolean;
}

export default function RepositorySelector({
  onSelect,
  selectedRepository,
  authenticated
}: RepositorySelectorProps) {
  const [repositories, setRepositories] = useState<GitHubRepository[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedSearchQuery] = useDebounce(searchQuery, 300);
  const [filterType, setFilterType] = useState<'all' | 'public' | 'private'>('all');
  const [sortBy, setSortBy] = useState<'name' | 'updated'>('updated');

  const loadRepositories = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const repos = await githubAPI.getRepositories();
      setRepositories(repos);
      toast.success(`${repos.length} repositorio(s) cargado(s)`);
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Error al cargar repositorios';
      setError(errorMessage);
      toast.error('Error al cargar repositorios', {
        description: errorMessage,
      });
      console.error('Error loading repositories:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (authenticated) {
      loadRepositories();
    }
  }, [authenticated, loadRepositories]);

  const filteredAndSortedRepos = useMemo(() => {
    let filtered = repositories;

    // Filtrar por búsqueda (usando debounced query)
    if (debouncedSearchQuery) {
      const query = debouncedSearchQuery.toLowerCase();
      filtered = filtered.filter(
        repo =>
          repo.name.toLowerCase().includes(query) ||
          repo.full_name.toLowerCase().includes(query) ||
          (repo.description?.toLowerCase().includes(query) ?? false)
      );
    }

    // Filtrar por tipo
    if (filterType === 'public') {
      filtered = filtered.filter(repo => !repo.private);
    } else if (filterType === 'private') {
      filtered = filtered.filter(repo => repo.private);
    }

    // Ordenar
    if (sortBy === 'name') {
      filtered = [...filtered].sort((a, b) => a.name.localeCompare(b.name));
    } else {
      filtered = [...filtered].sort(
        (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
      );
    }

    return filtered;
  }, [repositories, debouncedSearchQuery, filterType, sortBy]);

  if (!authenticated) {
    return (
      <div className="p-4 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300 text-center">
        <p className="text-gray-500">Conecta con GitHub para ver tus repositorios</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6" role="region" aria-labelledby="repository-selector-heading">
      <div className="flex items-center justify-between mb-4">
        <h2 id="repository-selector-heading" className="text-2xl font-semibold">Seleccionar Repositorio</h2>
        <button
          onClick={loadRepositories}
          disabled={loading}
          className="px-4 py-2 text-sm bg-gray-200 text-gray-700 rounded hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 transition-colors"
          aria-label={loading ? 'Cargando repositorios...' : 'Actualizar lista de repositorios'}
          type="button"
        >
          {loading ? 'Cargando...' : 'Actualizar'}
        </button>
      </div>

      {/* Filtros y búsqueda */}
      <div className="mb-4 space-y-3">
        <div className="relative">
          <label htmlFor="repository-search" className="sr-only">
            Buscar repositorios
          </label>
          <input
            id="repository-search"
            type="text"
            placeholder="Buscar repositorios..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            aria-label="Buscar repositorios por nombre o descripción"
          />
          <svg
            className="absolute right-3 top-2.5 w-5 h-5 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        </div>

        <div className="flex gap-4">
          <div className="flex-1">
            <label htmlFor="filter-type" className="block text-sm font-medium mb-1">
              Tipo
            </label>
            <select
              id="filter-type"
              value={filterType}
              onChange={(e) => setFilterType(e.target.value as 'all' | 'public' | 'private')}
              className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              aria-label="Filtrar repositorios por tipo"
            >
              <option value="all">Todos</option>
              <option value="public">Públicos</option>
              <option value="private">Privados</option>
            </select>
          </div>

          <div className="flex-1">
            <label htmlFor="sort-by" className="block text-sm font-medium mb-1">
              Ordenar por
            </label>
            <select
              id="sort-by"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'name' | 'updated')}
              className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              aria-label="Ordenar repositorios"
            >
              <option value="updated">Última actualización</option>
              <option value="name">Nombre</option>
            </select>
          </div>
        </div>
      </div>

      {/* Lista de repositorios */}
      {loading && repositories.length === 0 ? (
        <div className="flex items-center justify-center py-12" role="status" aria-label="Cargando repositorios">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" aria-hidden="true"></div>
          <span className="sr-only">Cargando repositorios...</span>
        </div>
      ) : error ? (
        <div 
          className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700"
          role="alert"
          aria-live="assertive"
        >
          {error}
        </div>
      ) : filteredAndSortedRepos.length === 0 ? (
        <div className="p-8 text-center text-gray-500" role="status">
          {searchQuery ? 'No se encontraron repositorios' : 'No hay repositorios disponibles'}
        </div>
      ) : (
        <ul className="space-y-2 max-h-96 overflow-y-auto" role="listbox" aria-label="Lista de repositorios">
          {filteredAndSortedRepos.map((repo) => (
            <li
              key={repo.id}
              role="option"
              aria-selected={selectedRepository?.id === repo.id}
              onClick={() => onSelect(repo)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onSelect(repo);
                }
              }}
              tabIndex={0}
              className={cn(
                "p-4 border rounded-lg cursor-pointer transition-all focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2",
                selectedRepository?.id === repo.id
                  ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200'
                  : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50'
              )}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <h3 className="font-semibold text-lg">{repo.name}</h3>
                    {repo.private && (
                      <span className="px-2 py-0.5 text-xs bg-gray-200 text-gray-700 rounded">
                        Privado
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{repo.full_name}</p>
                  {repo.description && (
                    <p className="text-sm text-gray-500 mt-2">{repo.description}</p>
                  )}
                  <div className="flex items-center gap-4 mt-2 text-xs text-gray-400">
                    {repo.language && (
                      <span className="flex items-center gap-1">
                        <span className="w-3 h-3 rounded-full bg-blue-500"></span>
                        {repo.language}
                      </span>
                    )}
                    <span>
                      Actualizado: {format(new Date(repo.updated_at), 'dd MMM yyyy', { locale: es })}
                    </span>
                  </div>
                </div>
                {selectedRepository?.id === repo.id && (
                  <svg
                    className="w-6 h-6 text-blue-500"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}

      {filteredAndSortedRepos.length > 0 && (
        <div className="mt-4 text-sm text-gray-500 text-center">
          Mostrando {filteredAndSortedRepos.length} de {repositories.length} repositorios
        </div>
      )}
    </div>
  );
}

