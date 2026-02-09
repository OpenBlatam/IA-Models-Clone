'use client';

import { useState } from 'react';
import { Store, Download, Star, Search } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface MarketplaceItem {
  id: string;
  name: string;
  description: string;
  category: 'plugin' | 'theme' | 'template' | 'widget';
  author: string;
  rating: number;
  downloads: number;
  price: 'free' | 'paid';
  priceAmount?: number;
}

export default function Marketplace() {
  const [searchTerm, setSearchTerm] = useState('');
  const [category, setCategory] = useState<'all' | 'plugin' | 'theme' | 'template' | 'widget'>('all');
  const [items] = useState<MarketplaceItem[]>([
    {
      id: '1',
      name: 'Advanced Analytics Plugin',
      description: 'Plugin avanzado para análisis de datos',
      category: 'plugin',
      author: 'Robot AI Team',
      rating: 4.8,
      downloads: 1250,
      price: 'free',
    },
    {
      id: '2',
      name: 'Dark Pro Theme',
      description: 'Tema oscuro profesional',
      category: 'theme',
      author: 'Community',
      rating: 4.5,
      downloads: 890,
      price: 'free',
    },
    {
      id: '3',
      name: 'Industrial Template Pack',
      description: 'Pack de plantillas para uso industrial',
      category: 'template',
      author: 'Industrial Solutions',
      rating: 4.9,
      downloads: 560,
      price: 'paid',
      priceAmount: 29.99,
    },
    {
      id: '4',
      name: 'Performance Widget',
      description: 'Widget para monitoreo de rendimiento',
      category: 'widget',
      author: 'Dev Team',
      rating: 4.7,
      downloads: 1200,
      price: 'free',
    },
  ]);

  const filteredItems = items.filter((item) => {
    const matchesSearch = item.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = category === 'all' || item.category === category;
    return matchesSearch && matchesCategory;
  });

  const handleInstall = (item: MarketplaceItem) => {
    toast.success(`${item.name} instalado exitosamente`);
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'plugin':
        return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
      case 'theme':
        return 'bg-purple-500/20 text-purple-400 border-purple-500/50';
      case 'template':
        return 'bg-green-500/20 text-green-400 border-green-500/50';
      case 'widget':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Store className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Marketplace</h3>
        </div>

        {/* Search and Filters */}
        <div className="mb-6 space-y-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Buscar en el marketplace..."
              className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>
          <div className="flex gap-2 flex-wrap">
            {(['all', 'plugin', 'theme', 'template', 'widget'] as const).map((cat) => (
              <button
                key={cat}
                onClick={() => setCategory(cat)}
                className={`px-3 py-1 rounded-lg text-sm transition-colors ${
                  category === cat
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {cat === 'all' ? 'Todos' : cat}
              </button>
            ))}
          </div>
        </div>

        {/* Items Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredItems.map((item) => (
            <div
              key={item.id}
              className="p-4 bg-gray-700/50 rounded-lg border border-gray-600 hover:border-primary-500/50 transition-colors"
            >
              <div className="flex items-start justify-between mb-2">
                <h4 className="font-semibold text-white flex-1">{item.name}</h4>
                <span className={`px-2 py-1 rounded text-xs border ${getCategoryColor(item.category)}`}>
                  {item.category}
                </span>
              </div>
              <p className="text-sm text-gray-300 mb-3">{item.description}</p>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-1">
                  <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                  <span className="text-sm text-white">{item.rating}</span>
                </div>
                <span className="text-xs text-gray-400">{item.downloads} descargas</span>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-gray-400">Por: {item.author}</p>
                  <p className="text-sm font-semibold text-white mt-1">
                    {item.price === 'free' ? 'Gratis' : `$${item.priceAmount}`}
                  </p>
                </div>
                <button
                  onClick={() => handleInstall(item)}
                  className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Instalar
                </button>
              </div>
            </div>
          ))}
        </div>

        {filteredItems.length === 0 && (
          <div className="text-center py-12 text-gray-400">
            <Store className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No se encontraron items</p>
          </div>
        )}
      </div>
    </div>
  );
}


