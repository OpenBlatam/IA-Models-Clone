'use client';

import { useState, useEffect } from 'react';
import { Zap, Activity, TrendingUp, AlertCircle } from 'lucide-react';

export function MusicPerformance() {
  const [metrics, setMetrics] = useState({
    renderTime: 0,
    memoryUsage: 0,
    networkLatency: 0,
    cacheHitRate: 0,
  });

  useEffect(() => {
    const updateMetrics = () => {
      if (performance.memory) {
        setMetrics((prev) => ({
          ...prev,
          memoryUsage: Math.round(performance.memory.usedJSHeapSize / 1048576), // MB
        }));
      }

      // Simular otras métricas
      setMetrics((prev) => ({
        ...prev,
        renderTime: Math.random() * 100,
        networkLatency: Math.random() * 200,
        cacheHitRate: Math.random() * 100,
      }));
    };

    const interval = setInterval(updateMetrics, 5000);
    updateMetrics();

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (value: number, threshold: number) => {
    if (value < threshold * 0.7) return 'text-green-400';
    if (value < threshold) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Zap className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Rendimiento</h3>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-white/5 rounded-lg border border-white/10">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-blue-300" />
            <span className="text-sm text-gray-400">Tiempo Render</span>
          </div>
          <p className={`text-2xl font-bold ${getStatusColor(metrics.renderTime, 100)}`}>
            {metrics.renderTime.toFixed(0)}ms
          </p>
        </div>

        <div className="p-4 bg-white/5 rounded-lg border border-white/10">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-4 h-4 text-green-300" />
            <span className="text-sm text-gray-400">Memoria</span>
          </div>
          <p className={`text-2xl font-bold ${getStatusColor(metrics.memoryUsage, 100)}`}>
            {metrics.memoryUsage}MB
          </p>
        </div>

        <div className="p-4 bg-white/5 rounded-lg border border-white/10">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-yellow-300" />
            <span className="text-sm text-gray-400">Latencia Red</span>
          </div>
          <p className={`text-2xl font-bold ${getStatusColor(metrics.networkLatency, 200)}`}>
            {metrics.networkLatency.toFixed(0)}ms
          </p>
        </div>

        <div className="p-4 bg-white/5 rounded-lg border border-white/10">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-purple-300" />
            <span className="text-sm text-gray-400">Cache Hit</span>
          </div>
          <p className={`text-2xl font-bold ${getStatusColor(100 - metrics.cacheHitRate, 50)}`}>
            {metrics.cacheHitRate.toFixed(0)}%
          </p>
        </div>
      </div>

      {(metrics.renderTime > 100 || metrics.memoryUsage > 100) && (
        <div className="mt-4 p-3 bg-yellow-500/20 border border-yellow-500/30 rounded-lg flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-yellow-400" />
          <p className="text-sm text-yellow-300">
            Rendimiento degradado. Considera recargar la página.
          </p>
        </div>
      )}
    </div>
  );
}


