'use client';

import { useState } from 'react';
import { Modal } from './ui/Modal';
import { Task } from '../types/task';

interface CommitApprovalModalProps {
  task: Task;
  isOpen: boolean;
  onApprove: () => void;
  onReject: () => void;
  onClose: () => void;
}

export function CommitApprovalModal({ task, isOpen, onApprove, onReject, onClose }: CommitApprovalModalProps) {
  if (!task.pendingCommitApproval) {
    return null;
  }

  const { commitSha, commitUrl, commitMessage, branch } = task.pendingCommitApproval;

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Aprobar o Rechazar Commit"
      size="lg"
      closeOnOverlayClick={false}
    >
      <div className="space-y-6">
        <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-blue-700 mb-3">
            <strong>⚠️ El commit ya fue ejecutado en GitHub.</strong>
          </p>
          <p className="text-xs text-blue-600">
            Si rechazas el commit, puedes revertirlo manualmente en GitHub. Esta acción solo marca la tarea como rechazada.
          </p>
        </div>

        {/* Commit Info */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Mensaje de Commit:
          </label>
          <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
            <p className="text-sm text-gray-900">{commitMessage}</p>
          </div>
        </div>

        {/* Commit Details */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">
              Branch:
            </label>
            <p className="text-sm text-gray-900">{branch || 'main'}</p>
          </div>
          {commitSha && (
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Commit SHA:
              </label>
              <p className="text-sm text-gray-900 font-mono">{commitSha.substring(0, 7)}</p>
            </div>
          )}
        </div>

        {/* Commit Link */}
        {commitUrl && (
          <div>
            <a
              href={commitUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-blue-600 hover:text-blue-800 underline"
            >
              Ver commit en GitHub →
            </a>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center justify-end gap-3 pt-4 border-t border-gray-200">
          <button
            onClick={onReject}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            Rechazar Commit
          </button>
          <button
            onClick={onApprove}
            className="px-4 py-2 text-sm font-medium text-white bg-green-600 rounded-lg hover:bg-green-700 transition-colors"
          >
            Aprobar Commit
          </button>
        </div>
      </div>
    </Modal>
  );
}

