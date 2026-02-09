'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FiActivity, FiZap, FiCpu } from 'react-icons/fi';

interface PerformanceMetrics {
  loadTime: number;
  renderTime: number;
  memoryUsage: number;
  networkLatency: number;
}

export default function PerformanceMonitor() {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const measurePerformance = () => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      const paint = performance.getEntriesByType('paint');
      
      const loadTime = navigation ? navigation.loadEventEnd - navigation.fetchStart : 0;
      const renderTime = paint.find((p) => p.name === 'first-contentful-paint')?.startTime || 0;
      
      const memory = (performance as any).memory;
      const memoryUsage = memory ? (memory.usedJSHeapSize / 1048576).toFixed(2) : 0;

      setMetrics({
        loadTime: Math.round(loadTime),
        renderTime: Math.round(renderTime),
        memoryUsage: parseFloat(memoryUsage as string),
        networkLatency: 0, // Would need to measure API calls
      });
    };

    // Measure on mount
    setTimeout(measurePerformance, 1000);

    // Toggle with Ctrl+Shift+P
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'P') {
        setIsVisible((prev) => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, []);

  if (!isVisible || !metrics) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="fixed top-20 right-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl p-4 z-50 min-w-[200px]"
    >
      <div className="flex items-center gap-2 mb-3">
        <FiActivity size={18} className="text-primary-600" />
        <h4 className="font-semibold text-gray-900 dark:text-white text-sm">Rendimiento</h4>
      </div>
      <div className="space-y-2 text-xs">
        <div className="flex items-center justify-between">
          <span className="text-gray-600 dark:text-gray-400">Tiempo de Carga</span>
          <span className="font-mono text-gray-900 dark:text-white">{metrics.loadTime}ms</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-600 dark:text-gray-400">Primer Render</span>
          <span className="font-mono text-gray-900 dark:text-white">{metrics.renderTime}ms</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-600 dark:text-gray-400">Memoria</span>
          <span className="font-mono text-gray-900 dark:text-white">{metrics.memoryUsage}MB</span>
        </div>
      </div>
    </motion.div>
  );
}


