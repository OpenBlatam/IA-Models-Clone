'use client';

import { useState, useEffect } from 'react';
import { Zap, CheckCircle, AlertCircle } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Optimization {
  name: string;
  description: string;
  enabled: boolean;
  impact: 'high' | 'medium' | 'low';
}

export default function PerformanceOptimizer() {
  const [optimizations, setOptimizations] = useState<Optimization[]>([
    {
      name: 'Lazy Loading',
      description: 'Cargar componentes solo cuando se necesitan',
      enabled: true,
      impact: 'high',
    },
    {
      name: 'Image Optimization',
      description: 'Optimizar imágenes automáticamente',
      enabled: true,
      impact: 'high',
    },
    {
      name: 'Code Splitting',
      description: 'Dividir código en chunks más pequeños',
      enabled: true,
      impact: 'medium',
    },
    {
      name: 'Cache Aggressive',
      description: 'Cache más agresivo para recursos estáticos',
      enabled: false,
      impact: 'medium',
    },
    {
      name: 'Debounce Inputs',
      description: 'Aplicar debounce a inputs de búsqueda',
      enabled: true,
      impact: 'low',
    },
  ]);

  const [performanceScore, setPerformanceScore] = useState(95);

  useEffect(() => {
    const enabledCount = optimizations.filter((opt) => opt.enabled).length;
    const totalCount = optimizations.length;
    const score = Math.round((enabledCount / totalCount) * 100);
    setPerformanceScore(score);
  }, [optimizations]);

  const toggleOptimization = (index: number) => {
    setOptimizations((prev) => {
      const updated = [...prev];
      updated[index].enabled = !updated[index].enabled;
      return updated;
    });
    toast.success(
      `${optimizations[index].enabled ? 'Desactivada' : 'Activada'}: ${optimizations[index].name}`
    );
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high':
        return 'text-red-400';
      case 'medium':
        return 'text-yellow-400';
      case 'low':
        return 'text-green-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Zap className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Optimizador de Rendimiento</h3>
        </div>

        {/* Performance Score */}
        <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">Puntuación de Rendimiento</span>
            <span className={`text-2xl font-bold ${
              performanceScore >= 90 ? 'text-green-400' : 
              performanceScore >= 70 ? 'text-yellow-400' : 'text-red-400'
            }`}>
              {performanceScore}%
            </span>
          </div>
          <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all ${
                performanceScore >= 90 ? 'bg-green-500' : 
                performanceScore >= 70 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
              style={{ width: `${performanceScore}%` }}
            />
          </div>
        </div>

        {/* Optimizations List */}
        <div className="space-y-3">
          {optimizations.map((opt, index) => (
            <div
              key={index}
              className="p-4 bg-gray-700/50 rounded-lg border border-gray-600"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-semibold text-white">{opt.name}</h4>
                    <span className={`text-xs ${getImpactColor(opt.impact)}`}>
                      Impacto: {opt.impact}
                    </span>
                  </div>
                  <p className="text-sm text-gray-300">{opt.description}</p>
                </div>
                <button
                  onClick={() => toggleOptimization(index)}
                  className={`relative w-14 h-7 rounded-full transition-colors flex-shrink-0 ${
                    opt.enabled ? 'bg-primary-600' : 'bg-gray-600'
                  }`}
                >
                  <div
                    className={`absolute top-1 left-1 w-5 h-5 bg-white rounded-full transition-transform ${
                      opt.enabled ? 'translate-x-7' : 'translate-x-0'
                    }`}
                  />
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Recommendations */}
        <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/50 rounded-lg">
          <div className="flex items-start gap-2">
            <AlertCircle className="w-5 h-5 text-blue-400 mt-0.5" />
            <div>
              <h4 className="text-sm font-semibold text-blue-400 mb-1">Recomendaciones</h4>
              <ul className="text-xs text-gray-300 space-y-1">
                <li>• Activa todas las optimizaciones de alto impacto para mejor rendimiento</li>
                <li>• El cache agresivo puede mejorar la velocidad pero usar más memoria</li>
                <li>• Revisa regularmente la puntuación de rendimiento</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


