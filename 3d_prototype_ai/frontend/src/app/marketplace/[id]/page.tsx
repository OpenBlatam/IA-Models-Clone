'use client'

import { useQuery } from '@tanstack/react-query'
import { marketplaceService } from '@/services/marketplace.service'
import { formatCurrency, formatDate } from '@/lib/utils'
import { useParams } from 'next/navigation'
import { ShoppingCart, Eye, Heart } from 'lucide-react'

export default function MarketplaceDetailPage() {
  const params = useParams()
  const id = params.id as string

  const { data: listing, isLoading } = useQuery({
    queryKey: ['marketplace-listing', id],
    queryFn: () => marketplaceService.getListing(id),
    enabled: !!id,
  })

  if (isLoading) {
    return (
      <div className="min-h-screen">
        <Header />
        <div className="container mx-auto px-4 py-8">
          <div className="text-center">Cargando...</div>
        </div>
      </div>
    )
  }

  if (!listing) {
    return (
      <div className="min-h-screen">
        <Header />
        <div className="container mx-auto px-4 py-8">
          <div className="text-center text-red-600">Listing no encontrado</div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-lg p-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">{listing.title}</h1>
          <p className="text-lg text-gray-600 mb-6">{listing.description}</p>

          <div className="flex items-center justify-between mb-8">
            <div className="text-3xl font-bold text-primary-600">
              {formatCurrency(listing.price)}
            </div>
            <div className="flex items-center gap-6 text-gray-600">
              <div className="flex items-center gap-2">
                <Eye className="w-5 h-5" />
                {listing.views} vistas
              </div>
              <div className="flex items-center gap-2">
                <Heart className="w-5 h-5" />
                {listing.likes} likes
              </div>
            </div>
          </div>

          <div className="text-sm text-gray-500 mb-8">
            Publicado el {formatDate(listing.created_at)}
          </div>

          <button className="w-full bg-primary-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-primary-700 transition-colors flex items-center justify-center gap-2">
            <ShoppingCart className="w-5 h-5" />
            Comprar Ahora
          </button>
        </div>
      </div>
    </div>
  )
}



