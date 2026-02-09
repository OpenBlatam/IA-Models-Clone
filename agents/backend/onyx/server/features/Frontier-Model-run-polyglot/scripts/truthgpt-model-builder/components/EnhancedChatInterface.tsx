'use client'

import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'sonner'
import { 
  History, 
  FileText, 
  GitCompare, 
  Settings, 
  Sparkles,
  TrendingUp,
  Zap,
} from 'lucide-react'
import { useModelStore } from '@/store/modelStore'
import { useModelCreation } from '@/lib/hooks'
import { saveModelToHistory, getModelHistory } from '@/lib/storage'
import Message from './Message'
import ModelStatus from './ModelStatus'
import ModelHistory from './ModelHistory'
import ModelPreview from './ModelPreview'
import ModelTemplates from './ModelTemplates'
import ModelComparator from './ModelComparator'
import ArchitectureVisualizer from './ArchitectureVisualizer'
import ModelStats from './ModelStats'
import QuickActions from './QuickActions'
import WelcomeTour from './WelcomeTour'
import DraftRecovery from './DraftRecovery'
import SmartInput from './SmartInput'
import OptimizationReport from './OptimizationReport'
import { useKeyboardShortcuts } from '@/lib/hooks'

export default function EnhancedChatInterface() {
  const [input, setInput] = useState('')
  const [showHistory, setShowHistory] = useState(false)
  const [showTemplates, setShowTemplates] = useState(false)
  const [showComparator, setShowComparator] = useState(false)
  const [selectedModels, setSelectedModels] = useState<any[]>([])
  const [validation, setValidation] = useState<any>(null)
  const [modelHistory, setModelHistory] = useState<any[]>([])
  const [showTour, setShowTour] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const { messages, addMessage, currentModel, setCurrentModel, clearMessages } = useModelStore()

  const {
    isCreating,
    progress,
    currentStep,
    error,
    previewSpec,
    costEstimate,
    analyzeDescription,
    createModel,
    reset,
  } = useModelCreation({
    onSuccess: (modelId) => {
      toast.success('¡Modelo creado exitosamente!')
      // Reload history
      setModelHistory(getModelHistory())
    },
    onError: (err) => {
      toast.error(`Error: ${err.message}`)
    },
    onProgress: (prog, step) => {
      // Progress is handled by the hook
    },
  })

  // Load history on mount
  useEffect(() => {
    setModelHistory(getModelHistory())
  }, [])

  // Keyboard shortcuts
  useKeyboardShortcuts([
    {
      keys: ['Ctrl', 'K'],
      callback: () => {
        const textarea = document.querySelector('textarea') as HTMLTextAreaElement
        textarea?.focus()
      },
    },
    {
      keys: ['Ctrl', 'H'],
      callback: () => setShowHistory(!showHistory),
    },
    {
      keys: ['Ctrl', 'T'],
      callback: () => setShowTemplates(!showTemplates),
    },
    {
      keys: ['Ctrl', 'C'],
      callback: () => {
        if (modelHistory.length >= 2) {
          setShowComparator(true)
        }
      },
    },
  ])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handlePreview = async (description: string) => {
    try {
      const result = await analyzeDescription(description)
      setShowPreview(true)
      return result
    } catch (err) {
      toast.error('Error al analizar la descripción')
      return null
    }
  }

  const handleSubmit = async (description: string) => {
    if (!description.trim()) return

    // Add user message
    addMessage({
      id: Date.now().toString(),
      text: description,
      sender: 'user',
      timestamp: new Date(),
    })

    // Show preview first
    const analysis = await handlePreview(description)
    if (!analysis) return

    // Create model
    const modelName = description.slice(0, 50).replace(/[^a-z0-9]/gi, '-')
    await createModel(description, modelName)

    // Add assistant message
    addMessage({
      id: (Date.now() + 1).toString(),
      text: `Estoy creando tu modelo "${modelName}". Este proceso puede tomar unos minutos...`,
      sender: 'assistant',
      timestamp: new Date(),
    })
  }

  const handleTemplateSelect = (template: any) => {
    setInput(template.description)
    setShowTemplates(false)
  }

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <div className="border-b border-slate-700 bg-slate-800/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Sparkles className="w-6 h-6 text-blue-400" />
              <h1 className="text-xl font-bold text-white">TruthGPT Model Builder</h1>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowTemplates(true)}
                className="p-2 rounded-lg hover:bg-slate-700 transition-colors"
                title="Templates (Ctrl+T)"
              >
                <FileText className="w-5 h-5 text-slate-400" />
              </button>

              <button
                onClick={() => setShowComparator(true)}
                disabled={modelHistory.length < 2}
                className="p-2 rounded-lg hover:bg-slate-700 transition-colors disabled:opacity-50"
                title="Comparar modelos (Ctrl+C)"
              >
                <GitCompare className="w-5 h-5 text-slate-400" />
              </button>

              <button
                onClick={() => setShowHistory(true)}
                className="p-2 rounded-lg hover:bg-slate-700 transition-colors"
                title="Historial (Ctrl+H)"
              >
                <History className="w-5 h-5 text-slate-400" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-hidden flex">
        {/* Messages area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center px-4">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="max-w-2xl"
                >
                  <Sparkles className="w-16 h-16 text-blue-400 mx-auto mb-4" />
                  <h2 className="text-3xl font-bold text-white mb-4">
                    Crea tu modelo de IA personalizado
                  </h2>
                  <p className="text-slate-400 mb-8">
                    Describe el modelo que necesitas y nuestro sistema lo creará automáticamente,
                    optimizado para TruthGPT
                  </p>

                  <QuickActions onSelect={handleTemplateSelect} />
                </motion.div>
              </div>
            )}

            <AnimatePresence>
              {messages.map((message) => (
                <Message key={message.id} message={message} />
              ))}
            </AnimatePresence>

            {/* Current model status */}
            {currentModel && (
              <ModelStatus
                model={currentModel}
                progress={progress}
                currentStep={currentStep}
              />
            )}

            {/* Architecture visualizer */}
            {previewSpec && (
              <ArchitectureVisualizer spec={previewSpec} />
            )}

            {/* Optimization report */}
            {previewSpec && costEstimate && (
              <OptimizationReport
                optimization={{
                  improvements: [],
                  performance: {
                    estimatedTrainingTime: costEstimate.trainingTime || 0,
                    estimatedMemoryUsage: costEstimate.memoryUsage || 0,
                    estimatedAccuracy: 85,
                  },
                }}
              />
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input area */}
          <div className="border-t border-slate-700 bg-slate-800/50 backdrop-blur-sm p-4">
            <DraftRecovery onRestore={setInput} />
            <SmartInput
              value={input}
              onChange={setInput}
              onSubmit={handleSubmit}
              isLoading={isCreating}
              onValidationChange={setValidation}
              showSuggestions={true}
            />
          </div>
        </div>

        {/* Sidebar with stats */}
        <div className="w-80 border-l border-slate-700 bg-slate-800/30 backdrop-blur-sm p-4 overflow-y-auto">
          <ModelStats />
        </div>
      </div>

      {/* Modals */}
      <AnimatePresence>
        {showHistory && (
          <ModelHistory
            onClose={() => setShowHistory(false)}
            onSelectModel={(model) => {
              setSelectedModels([model])
              setShowHistory(false)
            }}
          />
        )}

        {showTemplates && (
          <ModelTemplates
            onClose={() => setShowTemplates(false)}
            onSelect={handleTemplateSelect}
          />
        )}

        {showComparator && (
          <ModelComparator
            onClose={() => setShowComparator(false)}
            models={selectedModels.length > 0 ? selectedModels : modelHistory.slice(0, 2)}
          />
        )}

        {showTour && (
          <WelcomeTour onClose={() => setShowTour(false)} />
        )}
      </AnimatePresence>
    </div>
  )
}


