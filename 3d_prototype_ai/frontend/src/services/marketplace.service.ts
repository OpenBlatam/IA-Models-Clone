import { apiClient } from '@/lib/api-client'
import type { MarketplaceListing, PaginatedResponse } from '@/types'

export const marketplaceService = {
  // Buscar listings
  searchListings: async (
    query?: string,
    page = 1,
    pageSize = 20
  ): Promise<PaginatedResponse<MarketplaceListing>> => {
    return apiClient.get<PaginatedResponse<MarketplaceListing>>(
      '/api/v1/marketplace/listings/search',
      { q: query, page, page_size: pageSize }
    )
  },

  // Obtener listing por ID
  getListing: async (id: string): Promise<MarketplaceListing> => {
    return apiClient.get<MarketplaceListing>(`/api/v1/marketplace/listings/${id}`)
  },

  // Crear listing
  createListing: async (data: {
    prototype_id: string
    title: string
    description: string
    price: number
  }): Promise<MarketplaceListing> => {
    return apiClient.post<MarketplaceListing>('/api/v1/marketplace/listings', data)
  },

  // Publicar listing
  publishListing: async (id: string): Promise<MarketplaceListing> => {
    return apiClient.post<MarketplaceListing>(
      `/api/v1/marketplace/listings/${id}/publish`
    )
  },

  // Obtener estadísticas del marketplace
  getStats: async (): Promise<unknown> => {
    return apiClient.get('/api/v1/marketplace/stats')
  },

  // Crear orden
  createOrder: async (listingId: string): Promise<unknown> => {
    return apiClient.post('/api/v1/marketplace/orders', { listing_id: listingId })
  },
}



