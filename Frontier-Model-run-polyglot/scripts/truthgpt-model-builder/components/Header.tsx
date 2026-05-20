'use client'

import { useState } from 'react'
import { Brain, Settings, Keyboard } from 'lucide-react'
import SettingsComponent from './Settings'
import KeyboardShortcuts from './KeyboardShortcuts'
import NotificationCenter from './NotificationCenter'

export default function Header() {
  const [showSettings, setShowSettings] = useState(false)
  const [showShortcuts, setShowShortcuts] = useState(false)
  const [notifications, setNotifications] = useState<any[]>([])

  const handleMarkAsRead = (id: string) => {
    setNotifications(prev => prev.map(n => n.id === id ? { ...n, read: true } : n))
  }

  const handleClearAll = () => {
    setNotifications([])
  }

  return (
    <header className="bg-slate-800/50 backdrop-blur-sm border-b border-slate-700">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Brain className="w-8 h-8 text-purple-400" />
            <div>
              <h1 className="text-2xl font-bold text-white">TruthGPT Model Builder</h1>
              <p className="text-sm text-slate-400">Crea modelos de IA adaptados y despliégalos en GitHub</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <NotificationCenter
              notifications={notifications}
              onMarkAsRead={handleMarkAsRead}
              onClearAll={handleClearAll}
            />
            <button
              onClick={() => setShowShortcuts(true)}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
              title="Atajos de teclado (Ctrl+?)"
            >
              <Keyboard className="w-5 h-5 text-slate-300" />
            </button>
            <button
              onClick={() => setShowSettings(true)}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
              title="Configuración"
            >
              <Settings className="w-5 h-5 text-slate-300" />
            </button>
          </div>
        </div>
      </div>
      <SettingsComponent isOpen={showSettings} onClose={() => setShowSettings(false)} />
      <KeyboardShortcuts isOpen={showShortcuts} onClose={() => setShowShortcuts(false)} />
    </header>
  )
}

