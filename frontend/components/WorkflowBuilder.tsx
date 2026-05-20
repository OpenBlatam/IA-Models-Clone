'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiWorkflow, FiPlus, FiX, FiArrowRight } from 'react-icons/fi';

interface WorkflowStep {
  id: string;
  type: 'generate' | 'review' | 'approve' | 'publish';
  name: string;
  config: any;
}

interface Workflow {
  id: string;
  name: string;
  steps: WorkflowStep[];
  createdAt: Date;
}

export default function WorkflowBuilder() {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [currentWorkflow, setCurrentWorkflow] = useState<Workflow | null>(null);

  const stepTypes = [
    { id: 'generate', name: 'Generar', icon: FiPlus },
    { id: 'review', name: 'Revisar', icon: FiArrowRight },
    { id: 'approve', name: 'Aprobar', icon: FiArrowRight },
    { id: 'publish', name: 'Publicar', icon: FiArrowRight },
  ];

  const addStep = (type: WorkflowStep['type']) => {
    if (!currentWorkflow) return;

    const step: WorkflowStep = {
      id: Date.now().toString(),
      type,
      name: stepTypes.find((s) => s.id === type)?.name || type,
      config: {},
    };

    setCurrentWorkflow({
      ...currentWorkflow,
      steps: [...currentWorkflow.steps, step],
    });
  };

  const saveWorkflow = () => {
    if (!currentWorkflow || currentWorkflow.name.trim() === '') return;

    const updated = workflows.find((w) => w.id === currentWorkflow.id)
      ? workflows.map((w) => (w.id === currentWorkflow.id ? currentWorkflow : w))
      : [...workflows, currentWorkflow];

    setWorkflows(updated);
    localStorage.setItem('bul_workflows', JSON.stringify(updated));
    setCurrentWorkflow(null);
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => {
          setIsOpen(true);
          const stored = localStorage.getItem('bul_workflows');
          if (stored) {
            setWorkflows(JSON.parse(stored).map((w: any) => ({
              ...w,
              createdAt: new Date(w.createdAt),
            })));
          }
        }}
        className="btn btn-secondary"
      >
        <FiWorkflow size={18} className="mr-2" />
        Workflows
      </button>
    );
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
        onClick={() => setIsOpen(false)}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-4xl w-full max-h-[90vh] flex flex-col"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="p-6 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white">
              Constructor de Workflows
            </h3>
            <button onClick={() => setIsOpen(false)} className="btn-icon">
              <FiX size={20} />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-6">
            {!currentWorkflow ? (
              <div className="space-y-4">
                <button
                  onClick={() => {
                    setCurrentWorkflow({
                      id: Date.now().toString(),
                      name: '',
                      steps: [],
                      createdAt: new Date(),
                    });
                  }}
                  className="btn btn-primary w-full"
                >
                  <FiPlus size={18} className="mr-2" />
                  Crear Nuevo Workflow
                </button>

                {workflows.length > 0 && (
                  <div className="space-y-2">
                    {workflows.map((workflow) => (
                      <div
                        key={workflow.id}
                        className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg flex items-center justify-between"
                      >
                        <div>
                          <h4 className="font-medium text-gray-900 dark:text-white">
                            {workflow.name}
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            {workflow.steps.length} pasos
                          </p>
                        </div>
                        <button
                          onClick={() => setCurrentWorkflow(workflow)}
                          className="btn btn-secondary text-sm"
                        >
                          Editar
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                <input
                  type="text"
                  value={currentWorkflow.name}
                  onChange={(e) =>
                    setCurrentWorkflow({ ...currentWorkflow, name: e.target.value })
                  }
                  placeholder="Nombre del workflow"
                  className="input w-full"
                />

                <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-3">
                    Pasos del Workflow
                  </h4>
                  <div className="flex gap-2 mb-4 flex-wrap">
                    {stepTypes.map((type) => (
                      <button
                        key={type.id}
                        onClick={() => addStep(type.id as WorkflowStep['type'])}
                        className="btn btn-secondary text-sm"
                      >
                        <type.icon size={16} className="mr-1" />
                        {type.name}
                      </button>
                    ))}
                  </div>

                  <div className="space-y-2">
                    {currentWorkflow.steps.map((step, index) => (
                      <div
                        key={step.id}
                        className="flex items-center gap-3 p-3 bg-white dark:bg-gray-800 rounded-lg"
                      >
                        <span className="text-sm font-medium text-gray-500 dark:text-gray-400 w-8">
                          {index + 1}
                        </span>
                        <span className="flex-1 text-gray-900 dark:text-white">{step.name}</span>
                        {index < currentWorkflow.steps.length - 1 && (
                          <FiArrowRight className="text-gray-400" />
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex gap-2">
                  <button onClick={saveWorkflow} className="btn btn-primary flex-1">
                    Guardar Workflow
                  </button>
                  <button
                    onClick={() => setCurrentWorkflow(null)}
                    className="btn btn-secondary"
                  >
                    Cancelar
                  </button>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}


