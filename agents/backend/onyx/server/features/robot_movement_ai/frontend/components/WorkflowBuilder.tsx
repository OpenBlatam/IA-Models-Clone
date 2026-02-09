'use client';

import { useState } from 'react';
import { Workflow, Plus, Play, Save, Trash2 } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface WorkflowStep {
  id: string;
  type: 'action' | 'condition' | 'delay';
  name: string;
  config: Record<string, any>;
}

interface Workflow {
  id: string;
  name: string;
  description: string;
  steps: WorkflowStep[];
  enabled: boolean;
}

export default function WorkflowBuilder() {
  const [workflows, setWorkflows] = useState<Workflow[]>([
    {
      id: '1',
      name: 'Secuencia de Inicio',
      description: 'Secuencia automática de inicio del robot',
      steps: [
        { id: '1', type: 'action', name: 'Conectar Robot', config: {} },
        { id: '2', type: 'delay', name: 'Esperar 2 segundos', config: { duration: 2000 } },
        { id: '3', type: 'action', name: 'Ir a Home', config: {} },
      ],
      enabled: true,
    },
  ]);
  const [showAdd, setShowAdd] = useState(false);
  const [newName, setNewName] = useState('');
  const [newDescription, setNewDescription] = useState('');

  const handleAdd = () => {
    if (!newName.trim()) {
      toast.error('El nombre del workflow es requerido');
      return;
    }

    const newWorkflow: Workflow = {
      id: Date.now().toString(),
      name: newName,
      description: newDescription,
      steps: [],
      enabled: false,
    };

    setWorkflows([...workflows, newWorkflow]);
    setNewName('');
    setNewDescription('');
    setShowAdd(false);
    toast.success('Workflow creado');
  };

  const handleDelete = (id: string) => {
    setWorkflows(workflows.filter((w) => w.id !== id));
    toast.success('Workflow eliminado');
  };

  const handleToggle = (id: string) => {
    setWorkflows((prev) =>
      prev.map((w) => (w.id === id ? { ...w, enabled: !w.enabled } : w))
    );
    toast.success('Workflow actualizado');
  };

  const handleRun = (id: string) => {
    toast.info('Ejecutando workflow...');
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Workflow className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Constructor de Workflows</h3>
          </div>
          <button
            onClick={() => setShowAdd(!showAdd)}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Nuevo Workflow
          </button>
        </div>

        {/* Add Form */}
        {showAdd && (
          <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <h4 className="text-sm font-medium text-white mb-3">Crear Nuevo Workflow</h4>
            <div className="space-y-3">
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Nombre del workflow"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
              <textarea
                value={newDescription}
                onChange={(e) => setNewDescription(e.target.value)}
                placeholder="Descripción"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                rows={2}
              />
              <div className="flex gap-2">
                <button
                  onClick={handleAdd}
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
                >
                  Crear
                </button>
                <button
                  onClick={() => {
                    setShowAdd(false);
                    setNewName('');
                    setNewDescription('');
                  }}
                  className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
                >
                  Cancelar
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Workflows List */}
        <div className="space-y-3">
          {workflows.map((workflow) => (
            <div
              key={workflow.id}
              className={`p-4 rounded-lg border ${
                workflow.enabled
                  ? 'bg-green-500/10 border-green-500/50'
                  : 'bg-gray-700/50 border-gray-600'
              }`}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-semibold text-white">{workflow.name}</h4>
                    <span className="px-2 py-0.5 bg-gray-700 text-gray-300 text-xs rounded">
                      {workflow.steps.length} pasos
                    </span>
                  </div>
                  <p className="text-sm text-gray-300 mb-2">{workflow.description}</p>
                  <div className="flex gap-2 flex-wrap">
                    {workflow.steps.map((step, index) => (
                      <span
                        key={step.id}
                        className="px-2 py-1 bg-gray-800 text-gray-300 text-xs rounded"
                      >
                        {index + 1}. {step.name}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleRun(workflow.id)}
                    className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors flex items-center gap-2"
                  >
                    <Play className="w-3 h-3" />
                    Ejecutar
                  </button>
                  <button
                    onClick={() => handleToggle(workflow.id)}
                    className={`px-3 py-1 text-sm rounded-lg transition-colors ${
                      workflow.enabled
                        ? 'bg-yellow-600 hover:bg-yellow-700 text-white'
                        : 'bg-green-600 hover:bg-green-700 text-white'
                    }`}
                  >
                    {workflow.enabled ? 'Desactivar' : 'Activar'}
                  </button>
                  <button
                    onClick={() => handleDelete(workflow.id)}
                    className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded-lg transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


