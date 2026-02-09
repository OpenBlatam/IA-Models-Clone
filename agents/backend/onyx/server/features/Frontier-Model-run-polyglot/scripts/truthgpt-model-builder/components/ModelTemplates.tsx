'use client'

import { motion } from 'framer-motion'
import { Template, Image, MessageSquare, TrendingUp, Brain, Zap, BarChart3 } from 'lucide-react'

interface Template {
  id: string
  name: string
  description: string
  category: string
  icon: any
  color: string
  example: string
}

const templates: Template[] = [
  {
    id: 'sentiment-analysis',
    name: 'Análisis de Sentimientos',
    description: 'Clasifica el sentimiento de textos (positivo, negativo, neutro)',
    category: 'NLP',
    icon: MessageSquare,
    color: 'from-blue-500 to-cyan-500',
    example: 'Un modelo para análisis de sentimientos en español que clasifique textos en positivo, negativo o neutro',
  },
  {
    id: 'image-classifier',
    name: 'Clasificador de Imágenes',
    description: 'Detecta y clasifica objetos en imágenes',
    category: 'Visión',
    icon: Image,
    color: 'from-purple-500 to-pink-500',
    example: 'Un modelo para clasificar imágenes y detectar objetos con alta precisión',
  },
  {
    id: 'time-series',
    name: 'Predicción Temporal',
    description: 'Predice valores futuros en series temporales',
    category: 'Time Series',
    icon: TrendingUp,
    color: 'from-green-500 to-emerald-500',
    example: 'Un modelo para predecir valores futuros en series temporales con datos históricos',
  },
  {
    id: 'text-generator',
    name: 'Generador de Texto',
    description: 'Genera texto automático basado en contexto',
    category: 'Generativo',
    icon: Brain,
    color: 'from-orange-500 to-red-500',
    example: 'Un modelo generativo para crear texto automático en español',
  },
  {
    id: 'spam-detector',
    name: 'Detector de Spam',
    description: 'Identifica correos o mensajes spam',
    category: 'Clasificación',
    icon: Zap,
    color: 'from-yellow-500 to-amber-500',
    example: 'Un modelo para detectar spam en emails con alta precisión',
  },
  {
    id: 'price-predictor',
    name: 'Predictor de Precios',
    description: 'Predice precios basado en características',
    category: 'Regresión',
    icon: BarChart3,
    color: 'from-indigo-500 to-purple-500',
    example: 'Un modelo para predecir precios de productos basado en sus características',
  },
]

interface ModelTemplatesProps {
  onSelectTemplate: (template: Template) => void
}

export default function ModelTemplates({ onSelectTemplate }: ModelTemplatesProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 mb-4">
        <Template className="w-5 h-5 text-purple-400" />
        <h3 className="text-lg font-bold text-white">Templates de Modelos</h3>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {templates.map((template, index) => {
          const Icon = template.icon
          return (
            <motion.button
              key={template.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              onClick={() => onSelectTemplate(template)}
              className="group text-left p-4 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700 rounded-lg transition-all"
            >
              <div className={`w-12 h-12 rounded-lg bg-gradient-to-r ${template.color} flex items-center justify-center mb-3 group-hover:scale-110 transition-transform`}>
                <Icon className="w-6 h-6 text-white" />
              </div>
              <h4 className="text-sm font-bold text-white mb-1">{template.name}</h4>
              <p className="text-xs text-slate-400 mb-2">{template.description}</p>
              <p className="text-xs text-purple-400 font-medium">{template.category}</p>
            </motion.button>
          )
        })}
      </div>
    </div>
  )
}


