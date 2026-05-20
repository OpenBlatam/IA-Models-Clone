'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FiTrendingUp, FiClock, FiFileText } from 'react-icons/fi';

export default function QuickStats() {
  const [stats, setStats] = useState({
    documentsGenerated: 0,
    totalTimeSaved: 0,
    averageQuality: 0,
  });

  useEffect(() => {
    // Load from localStorage
    const stored = localStorage.getItem('bul_quick_stats');
    if (stored) {
      setStats(JSON.parse(stored));
    }

    // Calculate stats from tasks
    const calculateStats = () => {
      const tasks = JSON.parse(localStorage.getItem('bul_tasks') || '[]');
      const completed = tasks.filter((t: any) => t.status === 'completed');
      
      setStats({
        documentsGenerated: completed.length,
        totalTimeSaved: completed.length * 30, // Assume 30 min saved per doc
        averageQuality: 4.5, // Mock value
      });

      localStorage.setItem('bul_quick_stats', JSON.stringify(stats));
    };

    calculateStats();
  }, []);

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card"
      >
        <div className="flex items-center gap-3 mb-2">
          <FiFileText size={24} className="text-primary-600" />
          <div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {stats.documentsGenerated}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Documentos Generados
            </div>
          </div>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="card"
      >
        <div className="flex items-center gap-3 mb-2">
          <FiClock size={24} className="text-green-600" />
          <div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {stats.totalTimeSaved}h
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Tiempo Ahorrado
            </div>
          </div>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="card"
      >
        <div className="flex items-center gap-3 mb-2">
          <FiTrendingUp size={24} className="text-purple-600" />
          <div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {stats.averageQuality.toFixed(1)}/5
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Calidad Promedio
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}


