'use client'

import { useQuery } from '@tanstack/react-query'
import { prototypeService } from '@/services/prototype.service'
import { PrototypeView } from '@/components/prototypes/PrototypeView'
import { useParams } from 'next/navigation'

export default function PrototypeDetailPage() {
  const params = useParams()
  const id = params.id as string

  const { data: prototype, isLoading, error } = useQuery({
    queryKey: ['prototype', id],
    queryFn: () => prototypeService.getById(id),
    enabled: !!id,
  })

  if (isLoading) {
    return (
      <div className="min-h-screen">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center">Cargando prototipo...</div>
        </div>
      </div>
    )
  }

  if (error || !prototype) {
    return (
      <div className="min-h-screen">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center text-red-600">Error al cargar el prototipo</div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <PrototypeView prototype={prototype} />
    </div>
  )
}

