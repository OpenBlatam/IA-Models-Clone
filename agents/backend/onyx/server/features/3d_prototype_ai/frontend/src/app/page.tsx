'use client'

import Link from 'next/link'
import { ArrowRight, Sparkles, Box, TrendingUp } from 'lucide-react'

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-secondary-50">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-16">
        <div className="text-center max-w-4xl mx-auto">
          <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6">
            Genera Prototipos 3D
            <span className="text-primary-600 block">con Inteligencia Artificial</span>
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            Sistema completo para crear prototipos 3D con materiales, modelos CAD,
            instrucciones de ensamblaje y opciones de presupuesto
          </p>
          <div className="flex gap-4 justify-center">
            <Link
              href="/prototypes/create"
              className="bg-primary-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-primary-700 transition-colors flex items-center gap-2"
            >
              Crear Prototipo
              <ArrowRight className="w-5 h-5" />
            </Link>
            <Link
              href="/dashboard"
              className="bg-white text-primary-600 px-8 py-3 rounded-lg font-semibold border-2 border-primary-600 hover:bg-primary-50 transition-colors"
            >
              Ver Dashboard
            </Link>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-8 mt-16">
          <div className="bg-white p-6 rounded-xl shadow-lg">
            <Sparkles className="w-12 h-12 text-primary-600 mb-4" />
            <h3 className="text-xl font-semibold mb-2">Generación Inteligente</h3>
            <p className="text-gray-600">
              Crea prototipos completos desde descripciones en lenguaje natural usando IA
            </p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow-lg">
            <Box className="w-12 h-12 text-primary-600 mb-4" />
            <h3 className="text-xl font-semibold mb-2">Modelos CAD</h3>
            <p className="text-gray-600">
              Genera modelos CAD por partes en formatos STL, STEP y OBJ
            </p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow-lg">
            <TrendingUp className="w-12 h-12 text-primary-600 mb-4" />
            <h3 className="text-xl font-semibold mb-2">Análisis Avanzado</h3>
            <p className="text-gray-600">
              Análisis de costos, viabilidad y comparación de prototipos
            </p>
          </div>
        </div>

        {/* Stats Section */}
        <div className="mt-16 bg-white rounded-xl shadow-lg p-8">
          <div className="grid md:grid-cols-4 gap-8 text-center">
            <div>
              <div className="text-3xl font-bold text-primary-600">250+</div>
              <div className="text-gray-600 mt-2">Endpoints API</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-primary-600">81</div>
              <div className="text-gray-600 mt-2">Sistemas Funcionales</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-primary-600">65K+</div>
              <div className="text-gray-600 mt-2">Líneas de Código</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-primary-600">100%</div>
              <div className="text-gray-600 mt-2">Enterprise Ready</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

