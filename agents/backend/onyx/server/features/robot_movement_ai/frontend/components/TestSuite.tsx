'use client';

import { useState } from 'react';
import { TestTube, Play, CheckCircle, XCircle, Clock } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Test {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'running' | 'passed' | 'failed';
  duration?: number;
}

export default function TestSuite() {
  const [tests, setTests] = useState<Test[]>([
    {
      id: '1',
      name: 'Test de Conexión',
      description: 'Verifica la conexión con el robot',
      status: 'pending',
    },
    {
      id: '2',
      name: 'Test de Movimiento',
      description: 'Prueba el movimiento básico del robot',
      status: 'pending',
    },
    {
      id: '3',
      name: 'Test de Posición',
      description: 'Verifica la precisión de posicionamiento',
      status: 'pending',
    },
    {
      id: '4',
      name: 'Test de Seguridad',
      description: 'Valida los límites de seguridad',
      status: 'pending',
    },
  ]);
  const [isRunning, setIsRunning] = useState(false);

  const handleRunTest = async (testId: string) => {
    setTests((prev) =>
      prev.map((t) => (t.id === testId ? { ...t, status: 'running' as const } : t))
    );

    // Simulate test execution
    await new Promise((resolve) => setTimeout(resolve, 2000));

    const passed = Math.random() > 0.3;
    setTests((prev) =>
      prev.map((t) =>
        t.id === testId
          ? {
              ...t,
              status: (passed ? 'passed' : 'failed') as const,
              duration: Math.floor(Math.random() * 1000) + 500,
            }
          : t
      )
    );

    toast[passed ? 'success' : 'error'](
      `Test ${passed ? 'pasado' : 'fallido'}: ${tests.find((t) => t.id === testId)?.name}`
    );
  };

  const handleRunAll = async () => {
    setIsRunning(true);
    toast.info('Ejecutando todos los tests...');

    for (const test of tests) {
      await handleRunTest(test.id);
      await new Promise((resolve) => setTimeout(resolve, 500));
    }

    setIsRunning(false);
    const passed = tests.filter((t) => t.status === 'passed').length;
    toast.success(`${passed}/${tests.length} tests pasados`);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'passed':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-400" />;
      case 'running':
        return <Clock className="w-5 h-5 text-yellow-400 animate-spin" />;
      default:
        return <Clock className="w-5 h-5 text-gray-400" />;
    }
  };

  const passedCount = tests.filter((t) => t.status === 'passed').length;
  const failedCount = tests.filter((t) => t.status === 'failed').length;
  const runningCount = tests.filter((t) => t.status === 'running').length;

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <TestTube className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Suite de Pruebas</h3>
          </div>
          <button
            onClick={handleRunAll}
            disabled={isRunning}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Play className="w-4 h-4" />
            Ejecutar Todos
          </button>
        </div>

        {/* Statistics */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <p className="text-sm text-gray-400 mb-1">Total</p>
            <p className="text-2xl font-bold text-white">{tests.length}</p>
          </div>
          <div className="p-4 bg-green-500/10 rounded-lg border border-green-500/50">
            <p className="text-sm text-green-400 mb-1">Pasados</p>
            <p className="text-2xl font-bold text-green-400">{passedCount}</p>
          </div>
          <div className="p-4 bg-red-500/10 rounded-lg border border-red-500/50">
            <p className="text-sm text-red-400 mb-1">Fallidos</p>
            <p className="text-2xl font-bold text-red-400">{failedCount}</p>
          </div>
          <div className="p-4 bg-yellow-500/10 rounded-lg border border-yellow-500/50">
            <p className="text-sm text-yellow-400 mb-1">Ejecutando</p>
            <p className="text-2xl font-bold text-yellow-400">{runningCount}</p>
          </div>
        </div>

        {/* Tests List */}
        <div className="space-y-3">
          {tests.map((test) => (
            <div
              key={test.id}
              className={`p-4 rounded-lg border ${
                test.status === 'passed'
                  ? 'bg-green-500/10 border-green-500/50'
                  : test.status === 'failed'
                  ? 'bg-red-500/10 border-red-500/50'
                  : test.status === 'running'
                  ? 'bg-yellow-500/10 border-yellow-500/50'
                  : 'bg-gray-700/50 border-gray-600'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3 flex-1">
                  {getStatusIcon(test.status)}
                  <div className="flex-1">
                    <h4 className="font-semibold text-white mb-1">{test.name}</h4>
                    <p className="text-sm text-gray-300">{test.description}</p>
                    {test.duration && (
                      <p className="text-xs text-gray-400 mt-1">
                        Duración: {test.duration}ms
                      </p>
                    )}
                  </div>
                </div>
                <button
                  onClick={() => handleRunTest(test.id)}
                  disabled={test.status === 'running' || isRunning}
                  className="px-3 py-1 bg-primary-600 hover:bg-primary-700 text-white text-sm rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  <Play className="w-3 h-3" />
                  Ejecutar
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


