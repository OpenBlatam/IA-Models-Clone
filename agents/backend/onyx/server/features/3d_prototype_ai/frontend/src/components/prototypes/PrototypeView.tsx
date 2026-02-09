'use client'

import { useState } from 'react'
import type { PrototypeResponse, Material, CADPart, AssemblyStep, BudgetOption } from '@/types'
import { formatCurrency, formatDate } from '@/lib/utils'
import { Package, DollarSign, Clock, AlertCircle, FileText, Box } from 'lucide-react'

interface PrototypeViewProps {
  prototype: PrototypeResponse
}

export function PrototypeView({ prototype }: PrototypeViewProps) {
  const [activeTab, setActiveTab] = useState<'materials' | 'cad' | 'assembly' | 'budget'>('materials')

  const tabs = [
    { id: 'materials' as const, label: 'Materiales', icon: Package },
    { id: 'cad' as const, label: 'Partes CAD', icon: Box },
    { id: 'assembly' as const, label: 'Ensamblaje', icon: FileText },
    { id: 'budget' as const, label: 'Presupuesto', icon: DollarSign },
  ]

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-lg p-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">{prototype.product_name}</h1>
        <p className="text-lg text-gray-600 mb-6">{prototype.product_description}</p>

        {/* Stats */}
        <div className="grid md:grid-cols-4 gap-4">
          <div className="bg-primary-50 p-4 rounded-lg">
            <div className="flex items-center gap-2 text-primary-700 mb-2">
              <DollarSign className="w-5 h-5" />
              <span className="font-semibold">Costo Total</span>
            </div>
            <div className="text-2xl font-bold text-primary-900">
              {formatCurrency(prototype.total_cost_estimate)}
            </div>
          </div>
          <div className="bg-secondary-50 p-4 rounded-lg">
            <div className="flex items-center gap-2 text-secondary-700 mb-2">
              <Clock className="w-5 h-5" />
              <span className="font-semibold">Tiempo</span>
            </div>
            <div className="text-2xl font-bold text-secondary-900">
              {prototype.estimated_build_time}
            </div>
          </div>
          <div className="bg-green-50 p-4 rounded-lg">
            <div className="flex items-center gap-2 text-green-700 mb-2">
              <AlertCircle className="w-5 h-5" />
              <span className="font-semibold">Dificultad</span>
            </div>
            <div className="text-2xl font-bold text-green-900 capitalize">
              {prototype.difficulty_level}
            </div>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="flex items-center gap-2 text-gray-700 mb-2">
              <Package className="w-5 h-5" />
              <span className="font-semibold">Materiales</span>
            </div>
            <div className="text-2xl font-bold text-gray-900">
              {prototype.materials.length}
            </div>
          </div>
        </div>

        <div className="mt-4 text-sm text-gray-500">
          Generado el {formatDate(prototype.generated_at)}
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-xl shadow-lg">
        <div className="border-b border-gray-200">
          <div className="flex overflow-x-auto">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-6 py-4 font-medium transition-colors border-b-2 ${
                    activeTab === tab.id
                      ? 'border-primary-600 text-primary-600'
                      : 'border-transparent text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  {tab.label}
                </button>
              )
            })}
          </div>
        </div>

        <div className="p-6">
          {/* Materials Tab */}
          {activeTab === 'materials' && (
            <div className="space-y-4">
              {prototype.materials.map((material, index) => (
                <MaterialCard key={index} material={material} />
              ))}
            </div>
          )}

          {/* CAD Parts Tab */}
          {activeTab === 'cad' && (
            <div className="grid md:grid-cols-2 gap-4">
              {prototype.cad_parts.map((part, index) => (
                <CADPartCard key={index} part={part} />
              ))}
            </div>
          )}

          {/* Assembly Tab */}
          {activeTab === 'assembly' && (
            <div className="space-y-4">
              {prototype.assembly_instructions.map((step, index) => (
                <AssemblyStepCard key={index} step={step} />
              ))}
            </div>
          )}

          {/* Budget Tab */}
          {activeTab === 'budget' && (
            <div className="grid md:grid-cols-2 gap-4">
              {prototype.budget_options.map((option, index) => (
                <BudgetOptionCard key={index} option={option} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function MaterialCard({ material }: { material: Material }) {
  return (
    <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start mb-2">
        <h3 className="font-semibold text-lg">{material.name}</h3>
        <span className="text-primary-600 font-bold">{formatCurrency(material.total_price)}</span>
      </div>
      <div className="text-sm text-gray-600 mb-2">
        Cantidad: {material.quantity} {material.unit}
      </div>
      <div className="text-sm text-gray-600 mb-2">
        Precio unitario: {formatCurrency(material.price_per_unit)} / {material.unit}
      </div>
      {material.sources.length > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <div className="text-xs font-semibold text-gray-500 mb-1">Fuentes:</div>
          <div className="flex flex-wrap gap-2">
            {material.sources.map((source, index) => (
              <a
                key={index}
                href={source.url || '#'}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs bg-gray-100 px-2 py-1 rounded hover:bg-gray-200 transition-colors"
              >
                {source.name}
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function CADPartCard({ part }: { part: CADPart }) {
  return (
    <div className="border border-gray-200 rounded-lg p-4">
      <div className="flex justify-between items-start mb-2">
        <h3 className="font-semibold">{part.part_name}</h3>
        <span className="text-xs bg-primary-100 text-primary-700 px-2 py-1 rounded">
          {part.cad_format}
        </span>
      </div>
      <p className="text-sm text-gray-600 mb-2">{part.description}</p>
      <div className="text-xs text-gray-500">
        Material: {part.material} | Cantidad: {part.quantity}
      </div>
    </div>
  )
}

function AssemblyStepCard({ step }: { step: AssemblyStep }) {
  return (
    <div className="border border-gray-200 rounded-lg p-4">
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0 w-10 h-10 bg-primary-100 text-primary-700 rounded-full flex items-center justify-center font-bold">
          {step.step_number}
        </div>
        <div className="flex-1">
          <p className="text-gray-900 mb-2">{step.description}</p>
          <div className="flex flex-wrap gap-2 text-sm">
            {step.parts_involved.length > 0 && (
              <span className="text-gray-600">
                Partes: {step.parts_involved.join(', ')}
              </span>
            )}
            {step.tools_needed.length > 0 && (
              <span className="text-gray-600">
                Herramientas: {step.tools_needed.join(', ')}
              </span>
            )}
            {step.time_estimate && (
              <span className="text-primary-600 font-medium">{step.time_estimate}</span>
            )}
            <span className="text-secondary-600 capitalize">{step.difficulty}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function BudgetOptionCard({ option }: { option: BudgetOption }) {
  return (
    <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start mb-2">
        <h3 className="font-semibold text-lg capitalize">{option.budget_level}</h3>
        <span className="text-primary-600 font-bold text-xl">
          {formatCurrency(option.total_cost)}
        </span>
      </div>
      <p className="text-sm text-gray-600 mb-2">{option.description}</p>
      <div className="text-sm text-gray-500 mb-2">
        Calidad: <span className="font-medium capitalize">{option.quality_level}</span>
      </div>
      {option.trade_offs.length > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <div className="text-xs font-semibold text-gray-500 mb-1">Compromisos:</div>
          <ul className="text-xs text-gray-600 list-disc list-inside">
            {option.trade_offs.map((tradeoff, index) => (
              <li key={index}>{tradeoff}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}



