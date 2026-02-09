'use client';

import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { Modal } from './ui/Modal';
import { Task } from '../types/task';
import { cn } from '../utils/cn';

interface CommitPreviewModalProps {
  task: Task;
  isOpen: boolean;
  onCommit: () => void;
  onClose: () => void;
  isLoading?: boolean;
}

type TabType = 'files' | 'steps';

interface FileAction {
  path: string;
  content: string;
  action: 'create' | 'update';
}

export function CommitPreviewModal({ 
  task, 
  isOpen, 
  onCommit, 
  onClose,
  isLoading = false 
}: CommitPreviewModalProps) {
  const [expandedFiles, setExpandedFiles] = useState<Set<string>>(new Set());
  const [selectedTab, setSelectedTab] = useState<TabType>('files');
  
  // Generar nombre de archivo basado en el ID de la tarea para mantener consistencia
  // Esto evita que se generen múltiples archivos cuando el contenido cambia
  const fileNameRef = useRef<string | null>(null);
  
  // Generar el nombre del archivo solo una vez cuando se abre el modal
  useEffect(() => {
    if (isOpen && !fileNameRef.current) {
      // Solo generar si no hay plan (si hay plan, no se usa este nombre)
      if (!task.pendingApproval?.actions?.length) {
        // Usar ID de tarea para nombre consistente - evita múltiples archivos
        fileNameRef.current = `generated-content-${task.id}.md`;
      }
    }
    // Resetear cuando se cierra el modal
    if (!isOpen) {
      fileNameRef.current = null;
    }
  }, [isOpen, task.pendingApproval?.actions?.length, task.id]);

  const { plan, commitMessage, actions, hasPlan, hasContent } = useMemo(() => {
    // Si hay plan aprobado, usarlo
    if (task.pendingApproval?.approved && task.pendingApproval.actions?.length > 0) {
      return {
        plan: task.pendingApproval.plan,
        commitMessage: task.pendingApproval.commitMessage || 'feat: Cambios automáticos',
        actions: task.pendingApproval.actions || [],
        hasPlan: true,
        hasContent: true,
      };
    }
    
    // Si hay plan pero no está aprobado, usarlo de todas formas
    if (task.pendingApproval && task.pendingApproval.actions?.length > 0) {
      return {
        plan: task.pendingApproval.plan,
        commitMessage: task.pendingApproval.commitMessage || 'feat: Cambios automáticos',
        actions: task.pendingApproval.actions || [],
        hasPlan: true,
        hasContent: true,
      };
    }
    
    // Si no hay plan pero hay contenido generado, crear archivo .md (no JSON)
    const streamingContent = typeof task.streamingContent === 'string' 
      ? task.streamingContent 
      : task.streamingContent 
        ? (typeof task.streamingContent === 'object' 
            ? JSON.stringify(task.streamingContent, null, 2)
            : String(task.streamingContent))
        : '';
    
    // Crear archivo .md con el contenido generado
    // Usar el nombre de archivo estable del ref, no generar uno nuevo cada vez
    if (streamingContent && streamingContent.length > 0) {
      // Usar nombre basado en ID de tarea para consistencia
      const fileName = fileNameRef.current || `generated-content-${task.id}.md`;
      // Formatear como markdown, no como JSON
      const markdownContent = streamingContent.startsWith('```json') || streamingContent.startsWith('{')
        ? streamingContent.replace(/^```json\s*/, '').replace(/\s*```$/, '').trim()
        : streamingContent;
      
      return {
        plan: null,
        commitMessage: `feat: ${task.instruction?.substring(0, 50) || 'Contenido generado'}...`,
        actions: [{
          path: fileName,
          content: markdownContent,
          action: 'create' as const,
        }],
        hasPlan: false,
        hasContent: true,
      };
    }
    
    return { plan: null, commitMessage: '', actions: [], hasPlan: false, hasContent: false };
  }, [task.pendingApproval, task.streamingContent, task.instruction]);

  const { filesToCreate, filesToModify, totalCharacters } = useMemo(() => {
    const create = actions.filter((a): a is FileAction => a.action === 'create');
    const modify = actions.filter((a): a is FileAction => a.action === 'update');
    const totalChars = [...create, ...modify].reduce((sum, file) => sum + file.content.length, 0);
    return { filesToCreate: create, filesToModify: modify, totalCharacters: totalChars };
  }, [actions]);

  const toggleFile = useCallback((path: string) => {
    setExpandedFiles(prev => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  }, []);

  const handleTabChange = useCallback((tab: TabType) => {
    setSelectedTab(tab);
  }, []);

  // Si no hay contenido ni plan, no mostrar el modal
  if (!hasContent && !hasPlan) {
    return null;
  }

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Revisar Cambios - Confirmar Commit"
      size="xl"
      closeOnOverlayClick={false}
    >
      <div className="space-y-6">
        {hasPlan ? (
          <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
            <p className="text-sm text-green-700 mb-2">
              <strong>✅ Plan disponible - Listo para comitar</strong>
            </p>
            <p className="text-xs text-green-600">
              Revisa los cambios que se van a comitar. Una vez confirmes, se ejecutarán en GitHub.
            </p>
          </div>
        ) : (
          <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <p className="text-sm text-yellow-700 mb-2">
              <strong>⚠️ Sin plan estructurado - Se creará archivo .md</strong>
            </p>
            <p className="text-xs text-yellow-600">
              No hay un plan completo, pero hay contenido generado. Se creará un archivo .md con el contenido generado (no se creará JSON).
            </p>
          </div>
        )}

        {/* Commit Message */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Mensaje de Commit:
          </label>
          <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
            <p className="text-sm text-gray-900 font-mono">{commitMessage}</p>
          </div>
        </div>

        {/* Repository Info */}
        <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-blue-700">
                <strong>Repositorio:</strong> {task.repository}
              </p>
              {task.repoInfo?.default_branch && (
                <p className="text-xs text-blue-700 mt-1">
                  <strong>Rama:</strong> {task.repoInfo.default_branch}
                </p>
              )}
            </div>
            {totalCharacters > 0 && (
              <div className="text-right">
                <p className="text-xs font-semibold text-blue-700">
                  {totalCharacters.toLocaleString()} caracteres
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8" role="tablist">
            <button
              onClick={() => handleTabChange('files')}
              role="tab"
              aria-selected={selectedTab === 'files'}
              aria-controls="files-tabpanel"
              id="files-tab"
              className={cn(
                "py-2 px-1 border-b-2 font-medium text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 rounded-t",
                selectedTab === 'files'
                  ? "border-green-500 text-green-600"
                  : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
              )}
            >
              Archivos ({filesToCreate.length + filesToModify.length})
            </button>
            {plan?.steps && plan.steps.length > 0 && (
              <button
                onClick={() => handleTabChange('steps')}
                role="tab"
                aria-selected={selectedTab === 'steps'}
                aria-controls="steps-tabpanel"
                id="steps-tab"
                className={cn(
                  "py-2 px-1 border-b-2 font-medium text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 rounded-t",
                  selectedTab === 'steps'
                    ? "border-green-500 text-green-600"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                )}
              >
                Pasos del Plan ({plan.steps.length})
              </button>
            )}
          </nav>
        </div>

        {/* Content */}
        <div 
          className="max-h-96 overflow-y-auto"
          role="tabpanel"
          id={`${selectedTab}-tabpanel`}
          aria-labelledby={`${selectedTab}-tab`}
        >
          {selectedTab === 'files' && (
            <div className="space-y-4">
              {/* Total Characters Summary */}
              {totalCharacters > 0 && (
                <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
                  <p className="text-sm text-gray-700">
                    <strong>Total de contenido:</strong> {totalCharacters.toLocaleString()} caracteres
                  </p>
                </div>
              )}

              {/* Files to Create */}
              {filesToCreate.length > 0 && (
                <div>
                  <h4 className="text-sm font-semibold text-green-700 mb-2 flex items-center gap-2">
                    <span aria-hidden="true">➕</span>
                    Archivos a Crear ({filesToCreate.length})
                  </h4>
                  <div className="space-y-2">
                    {filesToCreate.map((file, idx) => {
                      const isExpanded = expandedFiles.has(file.path);
                      const lineCount = file.content.split('\n').length;
                      const charCount = file.content.length;
                      
                      return (
                        <div 
                          key={`create-${file.path}-${idx}`} 
                          className="border border-green-200 rounded-lg overflow-hidden"
                        >
                          <button
                            onClick={() => toggleFile(file.path)}
                            aria-expanded={isExpanded}
                            aria-controls={`file-content-${file.path}`}
                            className="w-full px-4 py-3 bg-green-50 hover:bg-green-100 flex items-center justify-between text-left transition-colors focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
                          >
                            <span className="text-sm font-medium text-green-900 font-mono truncate flex-1 mr-2">
                              {file.path}
                            </span>
                            <span className="text-xs text-green-600 whitespace-nowrap">
                              {charCount.toLocaleString()} caracteres
                            </span>
                            <svg 
                              className={cn(
                                "w-4 h-4 text-green-600 ml-2 transition-transform",
                                isExpanded && "transform rotate-180"
                              )}
                              fill="none" 
                              stroke="currentColor" 
                              viewBox="0 0 24 24"
                              aria-hidden="true"
                            >
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                            </svg>
                          </button>
                          {isExpanded && (
                            <div 
                              id={`file-content-${file.path}`}
                              className="p-4 bg-white border-t border-green-200"
                            >
                              <div className="mb-2 text-xs text-gray-600 flex items-center gap-4">
                                <span>{charCount.toLocaleString()} caracteres</span>
                                <span>•</span>
                                <span>{lineCount.toLocaleString()} líneas</span>
                              </div>
                              <div className="max-h-96 overflow-y-auto">
                                <pre className="text-xs overflow-x-auto bg-gray-50 p-3 rounded border font-mono">
                                  <code className="text-gray-800">{file.content}</code>
                                </pre>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Files to Modify */}
              {filesToModify.length > 0 && (
                <div>
                  <h4 className="text-sm font-semibold text-blue-700 mb-2 flex items-center gap-2">
                    <span aria-hidden="true">✏️</span>
                    Archivos a Modificar ({filesToModify.length})
                  </h4>
                  <div className="space-y-2">
                    {filesToModify.map((file, idx) => {
                      const isExpanded = expandedFiles.has(file.path);
                      const lineCount = file.content.split('\n').length;
                      const charCount = file.content.length;
                      
                      return (
                        <div 
                          key={`modify-${file.path}-${idx}`} 
                          className="border border-blue-200 rounded-lg overflow-hidden"
                        >
                          <button
                            onClick={() => toggleFile(file.path)}
                            aria-expanded={isExpanded}
                            aria-controls={`file-content-${file.path}`}
                            className="w-full px-4 py-3 bg-blue-50 hover:bg-blue-100 flex items-center justify-between text-left transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                          >
                            <span className="text-sm font-medium text-blue-900 font-mono truncate flex-1 mr-2">
                              {file.path}
                            </span>
                            <span className="text-xs text-blue-600 whitespace-nowrap">
                              {charCount.toLocaleString()} caracteres
                            </span>
                            <svg 
                              className={cn(
                                "w-4 h-4 text-blue-600 ml-2 transition-transform",
                                isExpanded && "transform rotate-180"
                              )}
                              fill="none" 
                              stroke="currentColor" 
                              viewBox="0 0 24 24"
                              aria-hidden="true"
                            >
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                            </svg>
                          </button>
                          {isExpanded && (
                            <div 
                              id={`file-content-${file.path}`}
                              className="p-4 bg-white border-t border-blue-200"
                            >
                              <div className="mb-2 text-xs text-gray-600 flex items-center gap-4">
                                <span>{charCount.toLocaleString()} caracteres</span>
                                <span>•</span>
                                <span>{lineCount.toLocaleString()} líneas</span>
                              </div>
                              <div className="max-h-96 overflow-y-auto">
                                <pre className="text-xs overflow-x-auto bg-gray-50 p-3 rounded border font-mono">
                                  <code className="text-gray-800">{file.content}</code>
                                </pre>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {filesToCreate.length === 0 && filesToModify.length === 0 && (
                <p className="text-sm text-gray-500 text-center py-8">
                  No hay archivos para crear o modificar
                </p>
              )}
            </div>
          )}

          {selectedTab === 'steps' && plan?.steps && (
            <div className="space-y-2">
              {plan.steps.map((step: string, idx: number) => (
                <div key={idx} className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
                  <div className="flex items-start gap-3">
                    <span className="flex-shrink-0 w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-xs font-medium">
                      {idx + 1}
                    </span>
                    <p className="text-sm text-gray-700 flex-1">{step}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Estado de Progreso */}
        {isLoading && (
          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center gap-3">
              <svg className="animate-spin h-5 w-5 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" aria-hidden="true">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <div>
                <p className="text-sm font-medium text-blue-700">
                  Ejecutando commit...
                </p>
                <p className="text-xs text-blue-600 mt-1">
                  Por favor espera, el proceso continuará sin detenerse
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center justify-end gap-3 pt-4 border-t border-gray-200">
          <button
            onClick={onClose}
            disabled={isLoading}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2"
          >
            {isLoading ? 'Procesando...' : 'Cerrar'}
          </button>
          <button
            onClick={onCommit}
            disabled={isLoading || !hasContent}
            className="px-4 py-2 text-sm font-medium text-white bg-green-600 rounded-lg hover:bg-green-700 transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
          >
            {isLoading ? (
              <>
                <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" aria-hidden="true">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Procesando...
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                Comitar
              </>
            )}
          </button>
        </div>
      </div>
    </Modal>
  );
}
