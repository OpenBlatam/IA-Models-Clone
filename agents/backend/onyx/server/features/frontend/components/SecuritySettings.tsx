'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiShield, FiLock, FiEye, FiEyeOff, FiKey } from 'react-icons/fi';

interface SecuritySettingsProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function SecuritySettings({ isOpen, onClose }: SecuritySettingsProps) {
  const [sessionTimeout, setSessionTimeout] = useState(30);
  const [requirePassword, setRequirePassword] = useState(false);
  const [twoFactor, setTwoFactor] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [password, setPassword] = useState('');

  const handleSave = () => {
    const settings = {
      sessionTimeout,
      requirePassword,
      twoFactor,
      password: requirePassword ? btoa(password) : null, // Simple encoding (not secure, just example)
    };
    localStorage.setItem('bul_security_settings', JSON.stringify(settings));
    alert('Configuración de seguridad guardada');
    onClose();
  };

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
            <FiShield size={20} className="text-primary-600" />
            <h3 className="font-semibold text-gray-900 dark:text-white">Seguridad</h3>
          </div>
          <button onClick={onClose} className="btn-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6L6 18" />
              <path d="M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Timeout de Sesión (minutos)
            </label>
            <input
              type="number"
              min="5"
              max="120"
              value={sessionTimeout}
              onChange={(e) => setSessionTimeout(parseInt(e.target.value))}
              className="input w-full"
            />
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              La sesión se cerrará automáticamente después de {sessionTimeout} minutos de inactividad
            </p>
          </div>

          <div>
            <label className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Requerir Contraseña
              </span>
              <input
                type="checkbox"
                checked={requirePassword}
                onChange={(e) => setRequirePassword(e.target.checked)}
                className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
              />
            </label>
            {requirePassword && (
              <div className="mt-3">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Contraseña
                </label>
                <div className="relative">
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Ingresa tu contraseña"
                    className="input w-full pr-10"
                  />
                  <button
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 btn-icon"
                  >
                    {showPassword ? <FiEyeOff size={18} /> : <FiEye size={18} />}
                  </button>
                </div>
              </div>
            )}
          </div>

          <div>
            <label className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Autenticación de Dos Factores
              </span>
              <input
                type="checkbox"
                checked={twoFactor}
                onChange={(e) => setTwoFactor(e.target.checked)}
                className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
              />
            </label>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Requiere un código adicional para iniciar sesión
            </p>
          </div>

          <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
            <div className="flex items-start gap-2">
              <FiLock size={18} className="text-yellow-600 dark:text-yellow-400 mt-0.5" />
              <div className="text-xs text-yellow-800 dark:text-yellow-200">
                <p className="font-medium mb-1">Nota de Seguridad</p>
                <p>
                  Estas configuraciones se guardan localmente. Para mayor seguridad, considera
                  implementar autenticación del lado del servidor.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-gray-200 dark:border-gray-700">
          <button onClick={handleSave} className="btn btn-primary w-full">
            <FiKey size={18} className="mr-2" />
            Guardar Configuración
          </button>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}


