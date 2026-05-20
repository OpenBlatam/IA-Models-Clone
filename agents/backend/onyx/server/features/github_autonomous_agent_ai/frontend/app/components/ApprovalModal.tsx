'use client';

import { useState } from 'react';
import { Modal } from './ui/Modal';
import { Task } from '../types/task';
import { cn } from '../utils/cn';

interface ApprovalModalProps {
  task: Task;
  isOpen: boolean;
  onApprove: () => void;
  onReject: () => void;
  onClose: () => void;
}

export function ApprovalModal({ task, isOpen, onApprove, onReject, onClose }: ApprovalModalProps) {
  const [expandedFiles, setExpandedFiles] = useState<Set<string>>(new Set());
  const [selectedTab, setSelectedTab] = useState<'files' | 'steps'>('files');

  if (!task.pendingApproval) {
    return null;
  }

  const { plan, commitMessage, actions } = task.pendingApproval;
  const filesToCreate = actions.filter(a => a.action === 'create');
  const filesToModify = actions.filter(a => a.action === 'update');

  const toggleFile = (path: string) => {
    setExpandedFiles(prev => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Revisar Plan - Aprobar para continuar"
      size="xl"
      closeOnOverlayClick={false}
    >
      <div className="space-y-6">
        <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-blue-700 mb-2">
            <strong>📋 Plan listo para revisar</strong>
          </p>
          <p className="text-xs text-blue-600">
            Revisa el plan y aprueba para continuar. Después de aprobar, el procesamiento continuará y podrás presionar el botón "Pausa" (⏸️) en la tarjeta de la tarea para ejecutar el commit.
          </p>
        </div>

        {/* Commit Message */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Mensaje de Commit:
          </label>
          <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
            <p className="text-sm text-gray-900">{commitMessage}</p>
          </div>
        </div>

        {/* Tabs */}
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setSelectedTab('files')}
              className={cn(
                "py-2 px-1 border-b-2 font-medium text-sm",
                selectedTab === 'files'
                  ? "border-blue-500 text-blue-600"
                  : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
              )}
            >
              Archivos ({filesToCreate.length + filesToModify.length})
            </button>
            {plan?.steps && plan.steps.length > 0 && (
              <button
                onClick={() => setSelectedTab('steps')}
                className={cn(
                  "py-2 px-1 border-b-2 font-medium text-sm",
                  selectedTab === 'steps'
                    ? "border-blue-500 text-blue-600"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                )}
              >
                Pasos del Plan ({plan.steps.length})
              </button>
            )}
          </nav>
        </div>

        {/* Content */}
        <div className="max-h-96 overflow-y-auto">
          {selectedTab === 'files' && (
            <div className="space-y-4">
              {/* Files to Create */}
              {filesToCreate.length > 0 && (
                <div>
                  <h4 className="text-sm font-semibold text-green-700 mb-2">
                    Archivos a Crear ({filesToCreate.length})
                  </h4>
                  <div className="space-y-2">
                    {filesToCreate.map((file, idx) => (
                      <div key={idx} className="border border-green-200 rounded-lg overflow-hidden">
                        <button
                          onClick={() => toggleFile(file.path)}
                          className="w-full px-4 py-3 bg-green-50 hover:bg-green-100 flex items-center justify-between text-left transition-colors"
                        >
                          <span className="text-sm font-medium text-green-900">{file.path}</span>
                          <span className="text-xs text-green-600">
                            {file.content.length} caracteres
                          </span>
                        </button>
                        {expandedFiles.has(file.path) && (
                          <div className="p-4 bg-white border-t border-green-200">
                            <div className="mb-2 text-xs text-gray-600">
                              {file.content.length} caracteres • {file.content.split('\n').length} líneas
                            </div>
                            <div className="max-h-96 overflow-y-auto">
                              <pre className="text-xs overflow-x-auto bg-gray-50 p-3 rounded border">
                                <code>{file.content}</code>
                              </pre>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Files to Modify */}
              {filesToModify.length > 0 && (
                <div>
                  <h4 className="text-sm font-semibold text-blue-700 mb-2">
                    Archivos a Modificar ({filesToModify.length})
                  </h4>
                  <div className="space-y-2">
                    {filesToModify.map((file, idx) => (
                      <div key={idx} className="border border-blue-200 rounded-lg overflow-hidden">
                        <button
                          onClick={() => toggleFile(file.path)}
                          className="w-full px-4 py-3 bg-blue-50 hover:bg-blue-100 flex items-center justify-between text-left transition-colors"
                        >
                          <span className="text-sm font-medium text-blue-900">{file.path}</span>
                          <span className="text-xs text-blue-600">
                            {file.content.length} caracteres
                          </span>
                        </button>
                        {expandedFiles.has(file.path) && (
                          <div className="p-4 bg-white border-t border-blue-200">
                            <div className="mb-2 text-xs text-gray-600">
                              {file.content.length} caracteres • {file.content.split('\n').length} líneas
                            </div>
                            <div className="max-h-96 overflow-y-auto">
                              <pre className="text-xs overflow-x-auto bg-gray-50 p-3 rounded border">
                                <code>{file.content}</code>
                              </pre>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
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
                    <span className="flex-shrink-0 w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-medium">
                      {idx + 1}
                    </span>
                    <p className="text-sm text-gray-700 flex-1">{step}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center justify-end gap-3 pt-4 border-t border-gray-200">
          <button
            onClick={onReject}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            Rechazar
          </button>
          <button
            onClick={onApprove}
            className="px-4 py-2 text-sm font-medium text-white bg-green-600 rounded-lg hover:bg-green-700 transition-colors"
          >
            Aprobar Plan
          </button>
        </div>
      </div>
    </Modal>
  );
}

