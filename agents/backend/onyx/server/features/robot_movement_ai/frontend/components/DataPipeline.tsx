'use client';

import { useState } from 'react';
import { GitBranch, Play, Pause, Settings } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface PipelineStage {
  id: string;
  name: string;
  type: 'source' | 'transform' | 'destination';
  status: 'pending' | 'running' | 'completed' | 'error';
}

interface Pipeline {
  id: string;
  name: string;
  stages: PipelineStage[];
  enabled: boolean;
}

export default function DataPipeline() {
  const [pipelines, setPipelines] = useState<Pipeline[]>([
    {
      id: '1',
      name: 'Pipeline de Datos',
      stages: [
        { id: '1', name: 'Recopilar Datos', type: 'source', status: 'completed' },
        { id: '2', name: 'Procesar', type: 'transform', status: 'running' },
        { id: '3', name: 'Almacenar', type: 'destination', status: 'pending' },
      ],
      enabled: true,
    },
  ]);

  const handleRun = (id: string) => {
    toast.info('Ejecutando pipeline...');
  };

  const handleToggle = (id: string) => {
    setPipelines((prev) =>
      prev.map((p) => (p.id === id ? { ...p, enabled: !p.enabled } : p))
    );
    toast.success('Pipeline actualizado');
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-500';
      case 'running':
        return 'bg-blue-500';
      case 'error':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'source':
        return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
      case 'transform':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
      case 'destination':
        return 'bg-green-500/20 text-green-400 border-green-500/50';
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <GitBranch className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Pipeline de Datos</h3>
        </div>

        {/* Pipelines */}
        <div className="space-y-4">
          {pipelines.map((pipeline) => (
            <div
              key={pipeline.id}
              className={`p-4 rounded-lg border ${
                pipeline.enabled
                  ? 'bg-green-500/10 border-green-500/50'
                  : 'bg-gray-700/50 border-gray-600'
              }`}
            >
              <div className="flex items-center justify-between mb-4">
                <h4 className="font-semibold text-white">{pipeline.name}</h4>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleRun(pipeline.id)}
                    className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors flex items-center gap-2"
                  >
                    <Play className="w-3 h-3" />
                    Ejecutar
                  </button>
                  <button
                    onClick={() => handleToggle(pipeline.id)}
                    className={`px-3 py-1 text-sm rounded-lg transition-colors ${
                      pipeline.enabled
                        ? 'bg-yellow-600 hover:bg-yellow-700 text-white'
                        : 'bg-green-600 hover:bg-green-700 text-white'
                    }`}
                  >
                    {pipeline.enabled ? 'Pausar' : 'Activar'}
                  </button>
                </div>
              </div>

              {/* Stages */}
              <div className="flex items-center gap-2">
                {pipeline.stages.map((stage, index) => (
                  <div key={stage.id} className="flex items-center gap-2 flex-1">
                    <div className="flex-1">
                      <div className={`p-3 rounded-lg border ${getTypeColor(stage.type)}`}>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs font-semibold">{stage.name}</span>
                          <div className={`w-2 h-2 rounded-full ${getStatusColor(stage.status)}`} />
                        </div>
                        <span className="text-xs text-gray-400 capitalize">{stage.type}</span>
                      </div>
                    </div>
                    {index < pipeline.stages.length - 1 && (
                      <div className="w-8 h-0.5 bg-gray-600" />
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


