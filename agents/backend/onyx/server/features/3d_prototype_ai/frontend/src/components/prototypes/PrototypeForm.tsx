'use client'

import { useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useMutation } from '@tanstack/react-query'
import { prototypeService } from '@/services/prototype.service'
import type { ProductType, PrototypeRequest } from '@/types'
import { Loader2, Sparkles } from 'lucide-react'
import toast from 'react-hot-toast'
import { useRouter } from 'next/navigation'

const prototypeSchema = z.object({
  product_description: z.string().min(10, 'La descripción debe tener al menos 10 caracteres'),
  product_type: z.nativeEnum(ProductType).optional(),
  budget: z.number().positive().optional(),
  requirements: z.array(z.string()).optional(),
  preferred_materials: z.array(z.string()).optional(),
  location: z.string().optional(),
})

type PrototypeFormData = z.infer<typeof prototypeSchema>

const productTypes = Object.values(ProductType)

export function PrototypeForm() {
  const router = useRouter()
  const [requirements, setRequirements] = useState<string[]>([])
  const [currentRequirement, setCurrentRequirement] = useState('')

  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
  } = useForm<PrototypeFormData>({
    resolver: zodResolver(prototypeSchema),
    defaultValues: {
      requirements: [],
    },
  })

  const mutation = useMutation({
    mutationFn: (data: PrototypeRequest) => prototypeService.generate(data),
    onSuccess: (data) => {
      toast.success('Prototipo generado exitosamente')
      router.push(`/prototypes/${data.product_name}`)
    },
    onError: (error: Error) => {
      toast.error(error.message || 'Error al generar el prototipo')
    },
  })

  const onSubmit = (data: PrototypeFormData) => {
    mutation.mutate({
      ...data,
      requirements: requirements.length > 0 ? requirements : undefined,
    })
  }

  const addRequirement = () => {
    if (currentRequirement.trim()) {
      setRequirements([...requirements, currentRequirement.trim()])
      setCurrentRequirement('')
    }
  }

  const removeRequirement = (index: number) => {
    setRequirements(requirements.filter((_, i) => i !== index))
  }

  return (
    <div className="max-w-3xl mx-auto p-6">
      <div className="bg-white rounded-xl shadow-lg p-8">
        <div className="flex items-center gap-3 mb-6">
          <Sparkles className="w-8 h-8 text-primary-600" />
          <h1 className="text-3xl font-bold text-gray-900">Crear Nuevo Prototipo</h1>
        </div>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          {/* Descripción del Producto */}
          <div>
            <label htmlFor="product_description" className="block text-sm font-medium text-gray-700 mb-2">
              Descripción del Producto *
            </label>
            <textarea
              {...register('product_description')}
              id="product_description"
              rows={4}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="Ej: Quiero hacer una nueva licuadora potente y fácil de limpiar"
            />
            {errors.product_description && (
              <p className="mt-1 text-sm text-red-600">{errors.product_description.message}</p>
            )}
          </div>

          {/* Tipo de Producto */}
          <div>
            <label htmlFor="product_type" className="block text-sm font-medium text-gray-700 mb-2">
              Tipo de Producto
            </label>
            <select
              {...register('product_type')}
              id="product_type"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="">Seleccionar tipo (opcional)</option>
              {productTypes.map((type) => (
                <option key={type} value={type}>
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </option>
              ))}
            </select>
          </div>

          {/* Presupuesto */}
          <div>
            <label htmlFor="budget" className="block text-sm font-medium text-gray-700 mb-2">
              Presupuesto (USD)
            </label>
            <input
              {...register('budget', { valueAsNumber: true })}
              type="number"
              id="budget"
              step="0.01"
              min="0"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="Ej: 150.00"
            />
          </div>

          {/* Requisitos */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Requisitos Adicionales
            </label>
            <div className="flex gap-2 mb-2">
              <input
                type="text"
                value={currentRequirement}
                onChange={(e) => setCurrentRequirement(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault()
                    addRequirement()
                  }
                }}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="Ej: Potente, Fácil de limpiar"
              />
              <button
                type="button"
                onClick={addRequirement}
                className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
              >
                Agregar
              </button>
            </div>
            {requirements.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {requirements.map((req, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center gap-2 px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm"
                  >
                    {req}
                    <button
                      type="button"
                      onClick={() => removeRequirement(index)}
                      className="hover:text-primary-900"
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Ubicación */}
          <div>
            <label htmlFor="location" className="block text-sm font-medium text-gray-700 mb-2">
              Ubicación
            </label>
            <input
              {...register('location')}
              type="text"
              id="location"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="Ej: México"
            />
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={mutation.isPending}
            className="w-full bg-primary-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {mutation.isPending ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Generando Prototipo...
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5" />
                Generar Prototipo
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  )
}



