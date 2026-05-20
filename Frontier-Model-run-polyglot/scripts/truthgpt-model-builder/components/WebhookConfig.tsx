'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Plus, Trash2, CheckCircle, AlertCircle, Webhook as WebhookIcon } from 'lucide-react'
import { toast } from 'react-hot-toast'
import { WebhookConfig, WebhookEvent, getWebhookManager } from '@/lib/webhook-manager'

interface WebhookConfigProps {
  onClose: () => void
}

const WEBHOOK_EVENTS: { value: WebhookEvent; label: string }[] = [
  { value: 'model.completed', label: 'Modelo Completado' },
  { value: 'model.failed', label: 'Modelo Fallido' },
  { value: 'model.started', label: 'Modelo Iniciado' },
  { value: 'batch.completed', label: 'Batch Completado' },
  { value: 'queue.empty', label: 'Cola Vacía' },
  { value: 'error.occurred', label: 'Error Ocurrido' },
]

export default function WebhookConfigPanel({ onClose }: WebhookConfigProps) {
  const webhookManager = getWebhookManager()
  const [webhooks, setWebhooks] = useState(webhookManager.getAllWebhooks())
  const [showAddForm, setShowAddForm] = useState(false)
  const [newWebhook, setNewWebhook] = useState<Partial<WebhookConfig>>({
    url: '',
    events: [],
    enabled: true,
    retries: 3,
    timeout: 5000,
  })

  const addWebhook = () => {
    if (!newWebhook.url || newWebhook.events?.length === 0) {
      toast.error('URL y al menos un evento son requeridos')
      return
    }

    try {
      const id = `webhook-${Date.now()}`
      webhookManager.registerWebhook(id, {
        url: newWebhook.url!,
        events: newWebhook.events!,
        secret: newWebhook.secret,
        enabled: newWebhook.enabled ?? true,
        retries: newWebhook.retries ?? 3,
        timeout: newWebhook.timeout ?? 5000,
      })

      setWebhooks(webhookManager.getAllWebhooks())
      setNewWebhook({
        url: '',
        events: [],
        enabled: true,
        retries: 3,
        timeout: 5000,
      })
      setShowAddForm(false)
      toast.success('Webhook agregado', { icon: '✅' })
    } catch (error) {
      toast.error('Error al agregar webhook')
    }
  }

  const removeWebhook = (id: string) => {
    webhookManager.unregisterWebhook(id)
    setWebhooks(webhookManager.getAllWebhooks())
    toast('Webhook eliminado', { icon: '🗑️' })
  }

  const toggleWebhook = (id: string, enabled: boolean) => {
    webhookManager.setWebhookEnabled(id, enabled)
    setWebhooks(webhookManager.getAllWebhooks())
  }

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="bg-slate-800 rounded-lg border border-slate-700 w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-700">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-600/20 rounded-lg">
              <WebhookIcon className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">Configuración de Webhooks</h3>
              <p className="text-sm text-slate-400">Integraciones externas</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {/* Lista de Webhooks */}
          {webhooks.length > 0 ? (
            <div className="space-y-3">
              {webhooks.map(({ id, config }) => (
                <div key={id} className="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        {config.enabled ? (
                          <CheckCircle className="w-4 h-4 text-green-400" />
                        ) : (
                          <AlertCircle className="w-4 h-4 text-slate-400" />
                        )}
                        <span className="text-sm font-medium text-white">{config.url}</span>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {config.events.map(event => (
                          <span
                            key={event}
                            className="text-xs px-2 py-0.5 bg-purple-600/20 text-purple-400 rounded"
                          >
                            {WEBHOOK_EVENTS.find(e => e.value === event)?.label || event}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <label className="flex items-center gap-1 text-xs">
                        <input
                          type="checkbox"
                          checked={config.enabled}
                          onChange={(e) => toggleWebhook(id, e.target.checked)}
                          className="rounded"
                        />
                        <span className="text-slate-400">Activo</span>
                      </label>
                      <button
                        onClick={() => removeWebhook(id)}
                        className="p-1 hover:bg-red-600/20 rounded transition-colors"
                      >
                        <Trash2 className="w-4 h-4 text-red-400" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-slate-400 py-8">
              <WebhookIcon className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No hay webhooks configurados</p>
            </div>
          )}

          {/* Formulario de Agregar */}
          <AnimatePresence>
            {showAddForm && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="bg-slate-700/30 rounded-lg p-4 border border-slate-600 space-y-3"
              >
                <h4 className="text-sm font-semibold text-slate-300">Nuevo Webhook</h4>
                <input
                  type="url"
                  placeholder="URL del webhook (https://...)"
                  value={newWebhook.url || ''}
                  onChange={(e) => setNewWebhook({ ...newWebhook, url: e.target.value })}
                  className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-2 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm"
                />
                <input
                  type="text"
                  placeholder="Secret (opcional)"
                  value={newWebhook.secret || ''}
                  onChange={(e) => setNewWebhook({ ...newWebhook, secret: e.target.value })}
                  className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-2 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm"
                />
                <div>
                  <label className="text-xs text-slate-400 mb-2 block">Eventos</label>
                  <div className="grid grid-cols-2 gap-2">
                    {WEBHOOK_EVENTS.map(event => (
                      <label key={event.value} className="flex items-center gap-2 text-sm text-slate-300">
                        <input
                          type="checkbox"
                          checked={newWebhook.events?.includes(event.value)}
                          onChange={(e) => {
                            const events = newWebhook.events || []
                            if (e.target.checked) {
                              setNewWebhook({ ...newWebhook, events: [...events, event.value] })
                            } else {
                              setNewWebhook({ ...newWebhook, events: events.filter(e => e !== event.value) })
                            }
                          }}
                          className="rounded"
                        />
                        <span>{event.label}</span>
                      </label>
                    ))}
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <label className="text-xs text-slate-400">Retries:</label>
                    <input
                      type="number"
                      min="0"
                      max="10"
                      value={newWebhook.retries || 3}
                      onChange={(e) => setNewWebhook({ ...newWebhook, retries: parseInt(e.target.value) || 3 })}
                      className="w-20 px-2 py-1 bg-slate-700/50 border border-slate-600 rounded text-white text-sm"
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    <label className="text-xs text-slate-400">Timeout (ms):</label>
                    <input
                      type="number"
                      min="1000"
                      max="30000"
                      step="1000"
                      value={newWebhook.timeout || 5000}
                      onChange={(e) => setNewWebhook({ ...newWebhook, timeout: parseInt(e.target.value) || 5000 })}
                      className="w-24 px-2 py-1 bg-slate-700/50 border border-slate-600 rounded text-white text-sm"
                    />
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={addWebhook}
                    className="flex-1 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors text-white font-medium text-sm"
                  >
                    Agregar Webhook
                  </button>
                  <button
                    onClick={() => {
                      setShowAddForm(false)
                      setNewWebhook({
                        url: '',
                        events: [],
                        enabled: true,
                        retries: 3,
                        timeout: 5000,
                      })
                    }}
                    className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors text-white text-sm"
                  >
                    Cancelar
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Botón Agregar */}
          {!showAddForm && (
            <button
              onClick={() => setShowAddForm(true)}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors text-slate-300"
            >
              <Plus className="w-4 h-4" />
              <span>Agregar Webhook</span>
            </button>
          )}
        </div>
      </motion.div>
    </div>
  )
}










