import React, { useState } from 'react';
import { Modal } from './ui/Modal';
import { Button } from './ui/Button';
import { Input } from './ui/Input';
import { Select } from './ui/Select';
import type { CreateAgentRequest, TaskType } from '../types/agent';

interface CreateAgentModalProps {
  open: boolean;
  onClose: () => void;
  onCreate: (request: CreateAgentRequest) => Promise<void>;
}

const TASK_TYPES: { value: TaskType; label: string }[] = [
  { value: 'code_review', label: 'Code Review' },
  { value: 'bug_fix', label: 'Bug Fix' },
  { value: 'feature', label: 'Feature' },
  { value: 'refactor', label: 'Refactor' },
  { value: 'documentation', label: 'Documentation' },
  { value: 'test', label: 'Test' },
];

export const CreateAgentModal: React.FC<CreateAgentModalProps> = ({
  open,
  onClose,
  onCreate,
}) => {
  const [formData, setFormData] = useState<CreateAgentRequest>({
    name: '',
    description: '',
    config: {
      taskType: 'code_review',
      frequency: 60,
      parameters: {},
      goal: '',
    },
  });
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.name.trim() || !formData.description.trim()) {
      return;
    }

    setIsSubmitting(true);
    try {
      await onCreate(formData);
      // Reset form
      setFormData({
        name: '',
        description: '',
        config: {
          taskType: 'code_review',
          frequency: 60,
          parameters: {},
          goal: '',
        },
      });
      onClose();
    } catch (error) {
      console.error('Error creating agent:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal
      isOpen={open}
      onClose={onClose}
      title="Crear Nuevo Agente"
      size="lg"
      closeOnOverlayClick={!isSubmitting}
    >
      <form onSubmit={handleSubmit} className="space-y-4">
            <Input
              label="Nombre del Agente"
              value={formData.name}
              onChange={(e) =>
                setFormData({ ...formData, name: e.target.value })
              }
              placeholder="Mi Agente"
              required
              fullWidth
            />

            <div>
              <label className="block text-sm font-medium text-black mb-1">
                Descripción
              </label>
              <textarea
                value={formData.description}
                onChange={(e) =>
                  setFormData({ ...formData, description: e.target.value })
                }
                placeholder="Descripción del agente..."
                className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-white text-black focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent"
                rows={3}
                required
              />
            </div>

            <Select
              label="Tipo de Tarea"
              value={formData.config.taskType}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  config: {
                    ...formData.config,
                    taskType: e.target.value as TaskType,
                  },
                })
              }
              options={TASK_TYPES.map((type) => ({
                value: type.value,
                label: type.label,
              }))}
              fullWidth
            />

            <Input
              type="number"
              label="Frecuencia (segundos)"
              value={formData.config.frequency.toString()}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  config: {
                    ...formData.config,
                    frequency: parseInt(e.target.value) || 60,
                  },
                })
              }
              min="1"
              fullWidth
            />

            <div>
              <label className="block text-sm font-medium text-black mb-1">
                Objetivo/Meta
              </label>
              <textarea
                value={formData.config.goal || ''}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    config: {
                      ...formData.config,
                      goal: e.target.value,
                    },
                  })
                }
                placeholder="Objetivo del agente..."
                className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-white text-black focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent"
                rows={4}
              />
            </div>

            <div className="flex gap-3 pt-4">
              <Button
                type="button"
                variant="ghost"
                onClick={onClose}
                fullWidth
              >
                Cancelar
              </Button>
              <Button
                type="submit"
                variant="primary"
                isLoading={isSubmitting}
                fullWidth
              >
                Crear Agente
              </Button>
            </div>
          </form>
    </Modal>
  );
};

