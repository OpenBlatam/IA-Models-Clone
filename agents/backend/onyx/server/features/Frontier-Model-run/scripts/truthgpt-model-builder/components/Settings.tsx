'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Settings as SettingsIcon, X, Moon, Sun, Bell, Download, Trash2 } from 'lucide-react'
import { savePreferences, getPreferences, clearModelHistory } from '@/lib/storage'
import { toast } from 'react-hot-toast'

interface SettingsProps {
  isOpen: boolean
  onClose: () => void
}

export default function Settings({ isOpen, onClose }: SettingsProps) {
  const [preferences, setPreferences] = useState(getPreferences())
  const [theme, setTheme] = useState(preferences.theme || 'dark')
  const [notifications, setNotifications] = useState(preferences.notifications !== false)

  const handleSave = () => {
    const newPrefs = {
      ...preferences,
      theme,
      notifications,
    }
    savePreferences(newPrefs)
    toast.success('Preferencias guardadas')
    onClose()
  }

  const handleClearHistory = () => {
    if (confirm('¿Estás seguro de eliminar todo el historial? Esta acción no se puede deshacer.')) {
      clearModelHistory()
      toast.success('Historial eliminado')
      onClose()
    }
  }

  if (!isOpen) return null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
          className="bg-slate-800 rounded-lg border border-slate-700 max-w-2xl w-full max-h-[90vh] overflow-y-auto"
        >
          <div className="sticky top-0 bg-slate-800/95 backdrop-blur-sm border-b border-slate-700 p-6 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <SettingsIcon className="w-6 h-6 text-purple-400" />
              <h2 className="text-xl font-bold text-white">Configuración</h2>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-slate-400" />
            </button>
          </div>

          <div className="p-6 space-y-6">
            {/* Theme */}
            <div>
              <h3 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
                {theme === 'dark' ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
                Tema
              </h3>
              <div className="flex gap-3">
                <button
                  onClick={() => setTheme('dark')}
                  className={`flex-1 p-3 rounded-lg border transition-all ${
                    theme === 'dark'
                      ? 'bg-slate-700 border-purple-500 text-white'
                      : 'bg-slate-700/50 border-slate-600 text-slate-400 hover:border-slate-500'
                  }`}
                >
                  <Moon className="w-5 h-5 mx-auto mb-2" />
                  <p className="text-sm font-medium">Oscuro</p>
                </button>
                <button
                  onClick={() => setTheme('light')}
                  className={`flex-1 p-3 rounded-lg border transition-all ${
                    theme === 'light'
                      ? 'bg-slate-700 border-purple-500 text-white'
                      : 'bg-slate-700/50 border-slate-600 text-slate-400 hover:border-slate-500'
                  }`}
                >
                  <Sun className="w-5 h-5 mx-auto mb-2" />
                  <p className="text-sm font-medium">Claro</p>
                </button>
              </div>
            </div>

            {/* Notifications */}
            <div>
              <h3 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
                <Bell className="w-4 h-4" />
                Notificaciones
              </h3>
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={notifications}
                  onChange={(e) => setNotifications(e.target.checked)}
                  className="w-5 h-5 rounded bg-slate-700 border-slate-600 text-purple-500 focus:ring-purple-500"
                />
                <span className="text-sm text-slate-300">
                  Recibir notificaciones cuando los modelos se completen
                </span>
              </label>
            </div>

            {/* Data Management */}
            <div>
              <h3 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
                <Download className="w-4 h-4" />
                Gestión de Datos
              </h3>
              <div className="space-y-2">
                <button
                  onClick={handleClearHistory}
                  className="w-full flex items-center gap-2 px-4 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 rounded-lg transition-colors text-sm text-red-400"
                >
                  <Trash2 className="w-4 h-4" />
                  <span>Eliminar todo el historial</span>
                </button>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-3 pt-4 border-t border-slate-700">
              <button
                onClick={onClose}
                className="flex-1 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
              >
                Cancelar
              </button>
              <button
                onClick={handleSave}
                className="flex-1 px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-lg transition-all font-medium"
              >
                Guardar
              </button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}


