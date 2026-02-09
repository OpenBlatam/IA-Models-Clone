'use client';

import { useState } from 'react';
import { useLocalStorage } from '@/lib/hooks/useLocalStorage';
import { FolderOpen, Plus, Trash2, Edit, Save } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Workspace {
  id: string;
  name: string;
  description: string;
  layout: any;
  createdAt: Date;
}

export default function WorkspaceManager() {
  const [workspaces, setWorkspaces] = useLocalStorage<Workspace[]>('workspaces', [
    {
      id: '1',
      name: 'Workspace Principal',
      description: 'Configuración principal de trabajo',
      layout: { tabs: ['control', 'chat', '3d'] },
      createdAt: new Date(),
    },
  ]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [newWorkspace, setNewWorkspace] = useState({ name: '', description: '' });

  const handleCreateWorkspace = () => {
    if (!newWorkspace.name.trim()) {
      toast.error('El nombre es requerido');
      return;
    }
    const workspace: Workspace = {
      id: Date.now().toString(),
      ...newWorkspace,
      layout: { tabs: [] },
      createdAt: new Date(),
    };
    setWorkspaces([...workspaces, workspace]);
    setNewWorkspace({ name: '', description: '' });
    toast.success('Workspace creado');
  };

  const handleDeleteWorkspace = (id: string) => {
    setWorkspaces(workspaces.filter((w) => w.id !== id));
    toast.success('Workspace eliminado');
  };

  const handleLoadWorkspace = (workspace: Workspace) => {
    // Would load workspace layout
    toast.success(`Workspace "${workspace.name}" cargado`);
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <FolderOpen className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Gestión de Workspaces</h3>
          </div>
        </div>

        {/* Create Workspace */}
        <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <h4 className="text-sm font-semibold text-white mb-3">Crear Nuevo Workspace</h4>
          <div className="space-y-3">
            <input
              type="text"
              value={newWorkspace.name}
              onChange={(e) => setNewWorkspace({ ...newWorkspace, name: e.target.value })}
              placeholder="Nombre del workspace"
              className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
            <textarea
              value={newWorkspace.description}
              onChange={(e) => setNewWorkspace({ ...newWorkspace, description: e.target.value })}
              placeholder="Descripción"
              rows={2}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500 resize-none"
            />
            <button
              onClick={handleCreateWorkspace}
              className="w-full px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Crear Workspace
            </button>
          </div>
        </div>

        {/* Workspaces List */}
        <div className="space-y-3">
          {workspaces.length === 0 ? (
            <div className="text-center py-12 text-gray-400">
              <FolderOpen className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No hay workspaces disponibles</p>
            </div>
          ) : (
            workspaces.map((workspace) => (
              <div
                key={workspace.id}
                className="p-4 bg-gray-700/50 rounded-lg border border-gray-600"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <h4 className="font-semibold text-white mb-1">{workspace.name}</h4>
                    <p className="text-sm text-gray-300 mb-2">{workspace.description}</p>
                    <p className="text-xs text-gray-400">
                      Creado: {workspace.createdAt.toLocaleDateString('es-ES')}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleLoadWorkspace(workspace)}
                      className="px-3 py-1 bg-primary-600 hover:bg-primary-700 text-white text-sm rounded transition-colors"
                    >
                      Cargar
                    </button>
                    <button
                      onClick={() => setEditingId(workspace.id)}
                      className="p-2 bg-gray-600 hover:bg-gray-700 text-white rounded transition-colors"
                      title="Editar"
                    >
                      <Edit className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleDeleteWorkspace(workspace.id)}
                      className="p-2 bg-red-600 hover:bg-red-700 text-white rounded transition-colors"
                      title="Eliminar"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}


