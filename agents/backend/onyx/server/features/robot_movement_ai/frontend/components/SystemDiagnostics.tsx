'use client';

import { useState } from 'react';
import { Stethoscope, Play, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Diagnostic {
  id: string;
  name: string;
  status: 'pass' | 'fail' | 'warning';
  message: string;
  duration: number;
}

export default function SystemDiagnostics() {
  const [diagnostics, setDiagnostics] = useState<Diagnostic[]>([
    {
      id: '1',
      name: 'Conexión de Red',
      status: 'pass',
      message: 'Conexión estable y funcionando correctamente',
      duration: 45,
    },
    {
      id: '2',
      name: 'Base de Datos',
      status: 'pass',
      message: 'Conexión a la base de datos exitosa',
      duration: 12,
    },
    {
      id: '3',
      name: 'Servicios Externos',
      status: 'warning',
      message: 'Algunos servicios externos tienen latencia elevada',
      duration: 250,
    },
    {
      id: '4',
      name: 'Sistema de Archivos',
      status: 'pass',
      message: 'Espacio en disco suficiente',
      duration: 8,
    },
  ]);
  const [isRunning, setIsRunning] = useState(false);

  const handleRunDiagnostics = async () => {
    setIsRunning(true);
    toast.info('Ejecutando diagnósticos...');

    // Simulate diagnostics
    await new Promise((resolve) => setTimeout(resolve, 2000));

    setDiagnostics((prev) =>
      prev.map((d) => ({
        ...d,
        status: Math.random() > 0.3 ? 'pass' : (Math.random() > 0.5 ? 'warning' : 'fail'),
        duration: Math.floor(Math.random() * 300) + 10,
      }))
    );

    setIsRunning(false);
    toast.success('Diagnósticos completados');
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pass':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
      case 'fail':
        return <XCircle className="w-5 h-5 text-red-400" />;
      default:
        return <CheckCircle className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pass':
        return 'bg-green-500/10 border-green-500/50';
      case 'warning':
        return 'bg-yellow-500/10 border-yellow-500/50';
      case 'fail':
        return 'bg-red-500/10 border-red-500/50';
      default:
        return 'bg-gray-700/50 border-gray-600';
    }
  };

  const passedCount = diagnostics.filter((d) => d.status === 'pass').length;
  const totalCount = diagnostics.length;
  const healthScore = (passedCount / totalCount) * 100;

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Stethoscope className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Diagnósticos del Sistema</h3>
          </div>
          <button
            onClick={handleRunDiagnostics}
            disabled={isRunning}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Play className="w-4 h-4" />
            {isRunning ? 'Ejecutando...' : 'Ejecutar Diagnósticos'}
          </button>
        </div>

        {/* Health Score */}
        <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-300">Puntuación de Salud</span>
            <span className="text-2xl font-bold text-white">{healthScore.toFixed(0)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-3">
            <div
              className={`h-3 rounded-full transition-all ${
                healthScore >= 80
                  ? 'bg-green-500'
                  : healthScore >= 50
                  ? 'bg-yellow-500'
                  : 'bg-red-500'
              }`}
              style={{ width: `${healthScore}%` }}
            />
          </div>
          <p className="text-xs text-gray-400 mt-2">
            {passedCount} de {totalCount} diagnósticos pasados
          </p>
        </div>

        {/* Diagnostics List */}
        <div className="space-y-3">
          {diagnostics.map((diagnostic) => (
            <div
              key={diagnostic.id}
              className={`p-4 rounded-lg border ${getStatusColor(diagnostic.status)}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3 flex-1">
                  {getStatusIcon(diagnostic.status)}
                  <div className="flex-1">
                    <h4 className="font-semibold text-white mb-1">{diagnostic.name}</h4>
                    <p className="text-sm text-gray-300">{diagnostic.message}</p>
                    <p className="text-xs text-gray-400 mt-1">
                      Duración: {diagnostic.duration}ms
                    </p>
                  </div>
                </div>
                <span className={`px-2 py-1 rounded text-xs capitalize ${
                  diagnostic.status === 'pass'
                    ? 'bg-green-500/20 text-green-400'
                    : diagnostic.status === 'warning'
                    ? 'bg-yellow-500/20 text-yellow-400'
                    : 'bg-red-500/20 text-red-400'
                }`}>
                  {diagnostic.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
