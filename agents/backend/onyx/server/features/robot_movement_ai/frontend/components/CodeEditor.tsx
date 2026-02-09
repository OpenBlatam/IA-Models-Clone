'use client';

import { useState } from 'react';
import { Code, Play, Save, FileCode } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

export default function CodeEditor() {
  const [code, setCode] = useState(`// Robot Movement Script
function moveRobot(x, y, z) {
  return {
    command: 'move',
    position: { x, y, z }
  };
}

// Example usage
const trajectory = [
  moveRobot(0.5, 0.3, 0.2),
  moveRobot(0.8, 0.5, 0.4),
  moveRobot(1.0, 0.7, 0.3)
];

// Execute trajectory
trajectory.forEach(point => {
  console.log('Moving to:', point.position);
});`);
  const [language, setLanguage] = useState('javascript');

  const handleRun = () => {
    try {
      // Simulate code execution
      toast.success('Código ejecutado exitosamente');
    } catch (error) {
      toast.error('Error al ejecutar código');
    }
  };

  const handleSave = () => {
    localStorage.setItem('robot_code', code);
    toast.success('Código guardado');
  };

  const handleLoad = () => {
    const saved = localStorage.getItem('robot_code');
    if (saved) {
      setCode(saved);
      toast.success('Código cargado');
    } else {
      toast.info('No hay código guardado');
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Code className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Editor de Código</h3>
          </div>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="px-3 py-1 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm"
          >
            <option value="javascript">JavaScript</option>
            <option value="python">Python</option>
            <option value="typescript">TypeScript</option>
          </select>
        </div>

        {/* Code Editor */}
        <div className="mb-4">
          <textarea
            value={code}
            onChange={(e) => setCode(e.target.value)}
            className="w-full h-96 p-4 bg-gray-900 border border-gray-600 rounded-lg text-white font-mono text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
            spellCheck={false}
          />
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <button
            onClick={handleRun}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Play className="w-4 h-4" />
            Ejecutar
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Save className="w-4 h-4" />
            Guardar
          </button>
          <button
            onClick={handleLoad}
            className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <FileCode className="w-4 h-4" />
            Cargar
          </button>
        </div>

        {/* Output */}
        <div className="mt-4 p-4 bg-gray-900 rounded-lg border border-gray-600">
          <h4 className="text-sm font-medium text-gray-300 mb-2">Salida:</h4>
          <div className="text-sm text-gray-400 font-mono">
            <p>Listo para ejecutar código...</p>
          </div>
        </div>
      </div>
    </div>
  );
}


