'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { marketplaceService } from '@/services/marketplace.service'
import { formatCurrency, formatDate } from '@/lib/utils'
import { Search, ShoppingCart, Eye, Heart } from 'lucide-react'
import Link from 'next/link'

export default function MarketplacePage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [page, setPage] = useState(1)

  const { data: listings, isLoading } = useQuery({
    queryKey: ['marketplace', searchQuery, page],
    queryFn: () => marketplaceService.searchListings(searchQuery, page, 20),
  })

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Marketplace</h1>
        </div>

        {/* Search */}
        <div className="mb-6">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Buscar prototipos..."
              className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
        </div>

        {isLoading ? (
          <div className="text-center py-12">Cargando...</div>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {listings?.items?.map((listing) => (
              <Link
                key={listing.id}
                href={`/marketplace/${listing.id}`}
                className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow"
              >
                <h2 className="text-xl font-semibold text-gray-900 mb-2">{listing.title}</h2>
                <p className="text-gray-600 mb-4 line-clamp-2">{listing.description}</p>
                <div className="flex items-center justify-between mb-4">
                  <div className="text-2xl font-bold text-primary-600">
                    {formatCurrency(listing.price)}
                  </div>
                  <div className="flex items-center gap-4 text-sm text-gray-500">
                    <div className="flex items-center gap-1">
                      <Eye className="w-4 h-4" />
                      {listing.views}
                    </div>
                    <div className="flex items-center gap-1">
                      <Heart className="w-4 h-4" />
                      {listing.likes}
                    </div>
                  </div>
                </div>
                <div className="text-xs text-gray-500">
                  {formatDate(listing.created_at)}
                </div>
              </Link>
            )) || (
              <div className="col-span-full text-center text-gray-500 py-12">
                No hay listings disponibles
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

