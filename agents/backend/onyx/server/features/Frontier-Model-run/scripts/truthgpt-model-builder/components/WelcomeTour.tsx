'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, ArrowRight, Sparkles } from 'lucide-react'
import { savePreferences, getPreferences } from '@/lib/storage'

interface TourStep {
  id: string
  title: string
  description: string
  target?: string
}

const tourSteps: TourStep[] = [
  {
    id: 'welcome',
    title: '¡Bienvenido a TruthGPT Model Builder!',
    description: 'Crea modelos de IA adaptados con solo describirlos. Te guiaré por las características principales.',
  },
  {
    id: 'input',
    title: 'Describe tu modelo',
    description: 'Escribe qué tipo de modelo necesitas. El sistema analizará tu descripción y creará la mejor arquitectura.',
    target: 'input-field',
  },
  {
    id: 'templates',
    title: 'Usa templates',
    description: 'Explora nuestros templates predefinidos para comenzar rápidamente.',
    target: 'templates-button',
  },
  {
    id: 'history',
    title: 'Revisa tu historial',
    description: 'Todos tus modelos se guardan automáticamente. Puedes verlos, exportarlos y compararlos.',
    target: 'history-button',
  },
]

interface WelcomeTourProps {
  onComplete: () => void
}

export default function WelcomeTour({ onComplete }: WelcomeTourProps) {
  const [currentStep, setCurrentStep] = useState(0)
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const prefs = getPreferences()
    if (!prefs.tourCompleted) {
      setIsVisible(true)
    }
  }, [])

  const handleNext = () => {
    if (currentStep < tourSteps.length - 1) {
      setCurrentStep(currentStep + 1)
    } else {
      handleComplete()
    }
  }

  const handleComplete = () => {
    savePreferences({ tourCompleted: true })
    setIsVisible(false)
    onComplete()
  }

  if (!isVisible) return null

  const step = tourSteps[currentStep]

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={handleComplete}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
          className="bg-slate-800 rounded-lg border border-slate-700 max-w-md w-full"
        >
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-purple-400" />
                <span className="text-sm text-slate-400">
                  {currentStep + 1} / {tourSteps.length}
                </span>
              </div>
              <button
                onClick={handleComplete}
                className="p-1 hover:bg-slate-700 rounded transition-colors"
              >
                <X className="w-4 h-4 text-slate-400" />
              </button>
            </div>

            <h3 className="text-lg font-bold text-white mb-2">{step.title}</h3>
            <p className="text-sm text-slate-300 mb-6">{step.description}</p>

            <div className="flex items-center justify-between">
              <button
                onClick={handleComplete}
                className="text-sm text-slate-400 hover:text-slate-300"
              >
                Omitir tour
              </button>
              <button
                onClick={handleNext}
                className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all text-sm font-medium"
              >
                {currentStep < tourSteps.length - 1 ? (
                  <>
                    Siguiente
                    <ArrowRight className="w-4 h-4" />
                  </>
                ) : (
                  'Comenzar'
                )}
              </button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}


