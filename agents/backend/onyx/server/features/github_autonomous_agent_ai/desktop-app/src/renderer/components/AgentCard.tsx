import React, { useState } from 'react';
import { Button } from './ui/Button';
import type { ContinuousAgent } from '../types/agent';

interface AgentCardProps {
  agent: ContinuousAgent;
  onToggle: (agentId: string) => void;
  onDelete: (agentId: string) => void;
  onEdit?: (agent: ContinuousAgent) => void;
}

export const AgentCard: React.FC<AgentCardProps> = ({
  agent,
  onToggle,
  onDelete,
  onEdit,
}) => {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [isToggling, setIsToggling] = useState(false);

  const handleToggle = async () => {
    setIsToggling(true);
    try {
      await onToggle(agent.id);
    } finally {
      setIsToggling(false);
    }
  };

  const handleDelete = () => {
    if (showDeleteConfirm) {
      onDelete(agent.id);
      setShowDeleteConfirm(false);
    } else {
      setShowDeleteConfirm(true);
    }
  };

  const statusColor = agent.isActive ? 'green' : 'gray';
  const statusText = agent.isActive ? 'Activo' : 'Inactivo';

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-sm transition-shadow">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h3 className="text-black text-xl font-normal mb-1">{agent.name}</h3>
          {agent.description && (
            <p className="text-sm text-gray-500">{agent.description}</p>
          )}
        </div>
        <div className="ml-4 flex items-center gap-2">
          <div
            className={`w-3 h-3 rounded-full ${
              agent.isActive ? 'bg-green-500' : 'bg-gray-400'
            }`}
          />
          <span className="text-sm text-black">{statusText}</span>
        </div>
      </div>
      <div>
          <div className="space-y-4">
            {/* Agent Config */}
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-500">Tipo:</span>
                <span className="ml-2 font-medium text-black capitalize">
                  {agent.config.taskType.replace('_', ' ')}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Frecuencia:</span>
                <span className="ml-2 font-medium text-black">
                  {agent.config.frequency}s
                </span>
              </div>
            </div>

            {/* Stats */}
            {agent.stats && (
              <div className="pt-4 border-t border-gray-200">
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <div className="text-gray-500">Ejecuciones</div>
                    <div className="font-semibold text-lg text-black">
                      {agent.stats.totalExecutions}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500">Exitosas</div>
                    <div className="font-semibold text-lg text-green-800">
                      {agent.stats.successfulExecutions}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500">Fallidas</div>
                    <div className="font-semibold text-lg text-red-800">
                      {agent.stats.failedExecutions}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-2 pt-4 border-t border-gray-200">
              <Button
                variant={agent.isActive ? 'danger' : 'primary'}
                size="sm"
                onClick={handleToggle}
                isLoading={isToggling}
                fullWidth
              >
                {agent.isActive ? 'Detener' : 'Iniciar'}
              </Button>
              {onEdit && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onEdit(agent)}
                  fullWidth
                >
                  Editar
                </Button>
              )}
              <Button
                variant="danger"
                size="sm"
                onClick={handleDelete}
                fullWidth
              >
                {showDeleteConfirm ? 'Confirmar' : 'Eliminar'}
              </Button>
            </div>

            {showDeleteConfirm && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
                ¿Estás seguro? Esta acción no se puede deshacer.
              </div>
            )}
          </div>
      </div>
    </div>
  );
};


