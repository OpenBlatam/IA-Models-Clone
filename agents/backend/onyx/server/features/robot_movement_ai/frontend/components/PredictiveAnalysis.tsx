'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api/client';
import { Brain, TrendingUp, AlertTriangle, Zap } from 'lucide-react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

export default function PredictiveAnalysis() {
  const [predictions, setPredictions] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [timeHorizon, setTimeHorizon] = useState(24); // hours

  useEffect(() => {
    loadPredictions();
  }, [timeHorizon]);

  const loadPredictions = async () => {
    setIsAnalyzing(true);
    try {
      // Simulate predictive analysis
      await new Promise((resolve) => setTimeout(resolve, 1500));

      const now = Date.now();
      const futureData = Array.from({ length: timeHorizon }, (_, i) => {
        const timestamp = now + i * 60 * 60 * 1000;
        return {
          time: new Date(timestamp).toLocaleTimeString('es-ES', {
            hour: '2-digit',
            minute: '2-digit',
          }),
          predicted: 50 + Math.random() * 30,
          confidence: 0.7 + Math.random() * 0.2,
          actual: i < 5 ? 50 + Math.random() * 30 : null,
        };
      });

      setPredictions({
        energy: {
          current: 75,
          predicted: 82,
          trend: 'up',
          confidence: 0.85,
        },
        performance: {
          current: 92,
          predicted: 88,
          trend: 'down',
          confidence: 0.78,
        },
        maintenance: {
          nextMaintenance: '2025-01-15',
          risk: 'low',
          confidence: 0.9,
        },
        timeSeries: futureData,
      });
    } catch (error) {
      console.error('Failed to load predictions:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Análisis Predictivo</h3>
          </div>
          <div className="flex items-center gap-3">
            <label className="text-sm text-gray-300">Horizonte:</label>
            <select
              value={timeHorizon}
              onChange={(e) => setTimeHorizon(parseInt(e.target.value))}
              className="px-3 py-1 bg-gray-700 border border-gray-600 rounded text-white text-sm"
            >
              <option value="6">6 horas</option>
              <option value="12">12 horas</option>
              <option value="24">24 horas</option>
              <option value="48">48 horas</option>
            </select>
          </div>
        </div>
      </div>

      {isAnalyzing ? (
        <div className="text-center py-12 text-gray-400">
          <Brain className="w-12 h-12 mx-auto mb-4 animate-pulse text-primary-400" />
          <p>Analizando datos y generando predicciones...</p>
        </div>
      ) : predictions ? (
        <>
          {/* Predictions Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h4 className="font-semibold text-white">Energía</h4>
                <Zap className="w-5 h-5 text-yellow-400" />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Actual</span>
                  <span className="text-white font-bold">{predictions.energy.current}%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Predicción</span>
                  <span
                    className={`font-bold ${
                      predictions.energy.trend === 'up' ? 'text-red-400' : 'text-green-400'
                    }`}
                  >
                    {predictions.energy.predicted}%
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Confianza</span>
                  <span className="text-white">
                    {(predictions.energy.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h4 className="font-semibold text-white">Rendimiento</h4>
                <TrendingUp className="w-5 h-5 text-green-400" />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Actual</span>
                  <span className="text-white font-bold">{predictions.performance.current}%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Predicción</span>
                  <span
                    className={`font-bold ${
                      predictions.performance.trend === 'up' ? 'text-green-400' : 'text-red-400'
                    }`}
                  >
                    {predictions.performance.predicted}%
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Confianza</span>
                  <span className="text-white">
                    {(predictions.performance.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h4 className="font-semibold text-white">Mantenimiento</h4>
                <AlertTriangle className="w-5 h-5 text-orange-400" />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Próximo</span>
                  <span className="text-white font-bold">
                    {new Date(predictions.maintenance.nextMaintenance).toLocaleDateString('es-ES')}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Riesgo</span>
                  <span
                    className={`font-bold ${
                      predictions.maintenance.risk === 'low'
                        ? 'text-green-400'
                        : predictions.maintenance.risk === 'medium'
                        ? 'text-yellow-400'
                        : 'text-red-400'
                    }`}
                  >
                    {predictions.maintenance.risk.toUpperCase()}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Confianza</span>
                  <span className="text-white">
                    {(predictions.maintenance.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Time Series Chart */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
            <h4 className="text-lg font-semibold text-white mb-4">
              Predicción de Métricas ({timeHorizon} horas)
            </h4>
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={predictions.timeSeries}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="predicted"
                  stroke="#0EA5E9"
                  fill="#0EA5E9"
                  fillOpacity={0.3}
                  name="Predicción"
                />
                <Area
                  type="monotone"
                  dataKey="actual"
                  stroke="#10B981"
                  fill="#10B981"
                  fillOpacity={0.3}
                  name="Actual"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </>
      ) : null}
    </div>
  );
}

