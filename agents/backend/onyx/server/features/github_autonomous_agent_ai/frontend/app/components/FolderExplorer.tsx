'use client';

import { useState, useEffect, useMemo, useCallback } from 'react';
import { githubAPI, GitHubContentItem, GitHubRepository } from '../lib/github-api';
import { cn } from '../utils/cn';

interface FolderExplorerProps {
  repository: GitHubRepository;
  onFolderSelect: (folderPath: string) => void;
  selectedFolder?: string;
}

export default function FolderExplorer({
  repository,
  onFolderSelect,
  selectedFolder,
}: FolderExplorerProps) {
  const [contents, setContents] = useState<GitHubContentItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPath, setCurrentPath] = useState<string>('');
  const [pathHistory, setPathHistory] = useState<string[]>([]);

  useEffect(() => {
    if (repository) {
      loadContents('');
    }
  }, [repository]);

  const loadContents = useCallback(async (path: string) => {
    try {
      setLoading(true);
      setError(null);
      
      const [owner, repo] = repository.full_name.split('/');
      const items = await githubAPI.getRepositoryContents(
        owner,
        repo,
        path,
        repository.default_branch
      );

      // Filtrar solo carpetas (directorios)
      const folders = items.filter(item => item.type === 'dir');
      setContents(folders);
      setCurrentPath(path);
    } catch (err: any) {
      setError(err.message || 'Error al cargar carpetas');
      console.error('Error loading contents:', err);
    } finally {
      setLoading(false);
    }
  }, [repository]);

  const handleFolderClick = useCallback((folder: GitHubContentItem) => {
    const newPath = folder.path;
    setPathHistory(prev => [...prev, currentPath]);
    loadContents(newPath);
  }, [currentPath, loadContents]);

  const handleBack = useCallback(() => {
    if (pathHistory.length > 0) {
      const previousPath = pathHistory[pathHistory.length - 1];
      setPathHistory(prev => prev.slice(0, -1));
      loadContents(previousPath);
    } else {
      loadContents('');
    }
  }, [pathHistory, loadContents]);

  const handleSelectFolder = useCallback((path: string) => {
    onFolderSelect(path);
  }, [onFolderSelect]);

  const breadcrumbs = useMemo(() => {
    if (!currentPath) return ['Raíz'];
    return ['Raíz', ...currentPath.split('/')];
  }, [currentPath]);

  return (
    <div className="bg-white rounded-lg shadow p-6" role="region" aria-labelledby="folder-explorer-heading">
      <div className="flex items-center justify-between mb-4">
        <h2 id="folder-explorer-heading" className="text-xl font-semibold">Explorar Carpetas</h2>
        {currentPath && (
          <button
            onClick={handleBack}
            className="px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors flex items-center gap-1 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
            aria-label="Volver a la carpeta anterior"
            type="button"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Atrás
          </button>
        )}
      </div>

      {/* Breadcrumbs */}
      <nav className="mb-4 flex items-center gap-2 text-sm text-gray-600 flex-wrap" aria-label="Ruta de navegación">
        {breadcrumbs.map((crumb, index) => (
          <div key={`${crumb}-${index}`} className="flex items-center gap-2">
            {index > 0 && <span aria-hidden="true">/</span>}
            <span className={index === breadcrumbs.length - 1 ? 'font-semibold text-black' : ''}>
              {crumb}
            </span>
          </div>
        ))}
      </nav>

      {error && (
        <div 
          className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm"
          role="alert"
          aria-live="assertive"
        >
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center py-8" role="status" aria-label="Cargando carpetas">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" aria-hidden="true"></div>
          <span className="sr-only">Cargando carpetas...</span>
        </div>
      ) : contents.length === 0 ? (
        <div className="text-center py-8 text-gray-500 text-sm" role="status">
          {currentPath ? 'No hay subcarpetas en esta ubicación' : 'No hay carpetas en la raíz del repositorio'}
        </div>
      ) : (
        <ul className="space-y-2 max-h-96 overflow-y-auto" role="list">
          {contents.map((item) => (
            <li
              key={item.sha}
              className="flex items-center justify-between p-3 border border-gray-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 transition-colors group"
            >
              <div className="flex items-center gap-3 flex-1">
                <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                </svg>
                <div className="flex-1">
                  <div className="font-medium text-sm text-black">{item.name}</div>
                  <div className="text-xs text-gray-500">{item.path}</div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleFolderClick(item)}
                  className="px-3 py-1.5 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                  aria-label={`Abrir carpeta ${item.name}`}
                  type="button"
                >
                  Abrir
                </button>
                <button
                  onClick={() => handleSelectFolder(item.path)}
                  className={cn(
                    "px-3 py-1.5 text-xs rounded transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2",
                    selectedFolder === item.path
                      ? 'bg-green-100 text-green-700 focus:ring-green-500'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200 focus:ring-gray-400'
                  )}
                  aria-label={selectedFolder === item.path ? `Carpeta ${item.name} seleccionada` : `Seleccionar carpeta ${item.name}`}
                  aria-pressed={selectedFolder === item.path}
                  type="button"
                >
                  {selectedFolder === item.path ? '✓ Seleccionada' : 'Seleccionar'}
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}

      {selectedFolder && (
        <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg" role="status" aria-live="polite">
          <p className="text-xs text-gray-600 mb-1">Carpeta seleccionada:</p>
          <p className="font-medium text-green-700 text-sm" aria-label={`Carpeta seleccionada: ${selectedFolder}`}>
            {selectedFolder}
          </p>
          <button
            onClick={() => handleSelectFolder('')}
            className="mt-2 text-xs text-red-600 hover:text-red-800 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 rounded"
            aria-label="Deseleccionar carpeta"
            type="button"
          >
            Deseleccionar
          </button>
        </div>
      )}
    </div>
  );
}



