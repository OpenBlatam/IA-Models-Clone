'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiInfo, FiX, FiMonitor, FiCpu, FiHardDrive } from 'react-icons/fi';

interface SystemInfoProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function SystemInfo({ isOpen, onClose }: SystemInfoProps) {
  const [info, setInfo] = useState({
    userAgent: '',
    platform: '',
    language: '',
    cookieEnabled: false,
    onLine: false,
    screenWidth: 0,
    screenHeight: 0,
    colorDepth: 0,
    timezone: '',
    memory: null as any,
  });

  useEffect(() => {
    if (isOpen) {
      setInfo({
        userAgent: navigator.userAgent,
        platform: navigator.platform,
        language: navigator.language,
        cookieEnabled: navigator.cookieEnabled,
        onLine: navigator.onLine,
        screenWidth: window.screen.width,
        screenHeight: window.screen.height,
        colorDepth: window.screen.colorDepth,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        memory: (performance as any).memory || null,
      });
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ x: 300 }}
        animate={{ x: 0 }}
        exit={{ x: 300 }}
        className="fixed right-0 top-0 h-full w-96 bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 shadow-xl z-40 flex flex-col"
      >
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FiInfo size={20} className="text-primary-600" />
            <h3 className="font-semibold text-gray-900 dark:text-white">Información del Sistema</h3>
          </div>
          <button onClick={onClose} className="btn-icon">
            <FiX size={20} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <FiMonitor size={16} className="text-gray-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Navegador</span>
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-700 p-2 rounded font-mono break-all">
              {info.userAgent}
            </div>
          </div>

          <div>
            <div className="flex items-center gap-2 mb-2">
              <FiCpu size={16} className="text-gray-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Plataforma</span>
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              {info.platform} • {info.language}
            </div>
          </div>

          <div>
            <div className="flex items-center gap-2 mb-2">
              <FiMonitor size={16} className="text-gray-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Pantalla</span>
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              {info.screenWidth} × {info.screenHeight} • {info.colorDepth} bits
            </div>
          </div>

          {info.memory && (
            <div>
              <div className="flex items-center gap-2 mb-2">
                <FiHardDrive size={16} className="text-gray-400" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Memoria</span>
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Usado: {(info.memory.usedJSHeapSize / 1048576).toFixed(2)} MB
                <br />
                Total: {(info.memory.totalJSHeapSize / 1048576).toFixed(2)} MB
                <br />
                Límite: {(info.memory.jsHeapSizeLimit / 1048576).toFixed(2)} MB
              </div>
            </div>
          )}

          <div>
            <div className="flex items-center gap-2 mb-2">
              <FiInfo size={16} className="text-gray-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Otros</span>
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <div>Zona horaria: {info.timezone}</div>
              <div>Cookies: {info.cookieEnabled ? 'Habilitadas' : 'Deshabilitadas'}</div>
              <div>Estado: {info.onLine ? 'En línea' : 'Sin conexión'}</div>
            </div>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}


