'use client'

import { useQuery } from '@tanstack/react-query'
import { prototypeService } from '@/services/prototype.service'
import { formatCurrency, formatDate } from '@/lib/utils'
import { Package, DollarSign, Calendar } from 'lucide-react'
import Link from 'next/link'

export default function PrototypesPage() {
  const { data: history, isLoading } = useQuery({
    queryKey: ['history', 1, 20],
    queryFn: () => prototypeService.getHistory(1, 20),
  })

  if (isLoading) {
    return (
      <div className="min-h-screen">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center">Cargando...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Mis Prototipos</h1>
          <Link
            href="/prototypes/create"
            className="bg-primary-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-primary-700 transition-colors"
          >
            Crear Nuevo
          </Link>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {history?.items?.map((prototype) => (
            <Link
              key={prototype.id}
              href={`/prototypes/${prototype.id}`}
              className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow"
            >
              <h2 className="text-xl font-semibold text-gray-900 mb-2">{prototype.product_name}</h2>
              <div className="flex items-center gap-2 text-sm text-gray-600 mb-4">
                <Package className="w-4 h-4" />
                <span className="capitalize">{prototype.product_type}</span>
              </div>
              <div className="flex items-center gap-2 text-primary-600 font-bold mb-2">
                <DollarSign className="w-4 h-4" />
                {formatCurrency(prototype.total_cost_estimate)}
              </div>
              <div className="flex items-center gap-2 text-xs text-gray-500">
                <Calendar className="w-4 h-4" />
                {formatDate(prototype.generated_at)}
              </div>
            </Link>
          )) || (
            <div className="col-span-full text-center text-gray-500 py-12">
              No hay prototipos aún. Crea tu primer prototipo!
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

