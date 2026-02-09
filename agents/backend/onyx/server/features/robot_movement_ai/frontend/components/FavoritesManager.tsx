'use client';

import { useState } from 'react';
import { useLocalStorage } from '@/lib/hooks/useLocalStorage';
import { Star, X } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

export default function FavoritesManager() {
  const [favorites, setFavorites] = useLocalStorage<string[]>('favorite-tabs', []);
  const [newFavorite, setNewFavorite] = useState('');

  const availableTabs = [
    'control',
    'chat',
    '3d',
    'status',
    'metrics',
    'history',
    'optimize',
    'recording',
  ];

  const handleAddFavorite = () => {
    if (!newFavorite.trim()) {
      toast.error('Por favor selecciona una pestaña');
      return;
    }
    if (favorites.includes(newFavorite)) {
      toast.error('Esta pestaña ya está en favoritos');
      return;
    }
    setFavorites([...favorites, newFavorite]);
    setNewFavorite('');
    toast.success('Favorito agregado');
  };

  const handleRemoveFavorite = (tab: string) => {
    setFavorites(favorites.filter((f) => f !== tab));
    toast.success('Favorito eliminado');
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Star className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Gestión de Favoritos</h3>
        </div>

        {/* Add Favorite */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Agregar Favorito
          </label>
          <div className="flex gap-2">
            <select
              value={newFavorite}
              onChange={(e) => setNewFavorite(e.target.value)}
              className="flex-1 px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="">Selecciona una pestaña</option>
              {availableTabs
                .filter((tab) => !favorites.includes(tab))
                .map((tab) => (
                  <option key={tab} value={tab}>
                    {tab.charAt(0).toUpperCase() + tab.slice(1)}
                  </option>
                ))}
            </select>
            <button
              onClick={handleAddFavorite}
              className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
            >
              Agregar
            </button>
          </div>
        </div>

        {/* Favorites List */}
        <div>
          <h4 className="text-sm font-medium text-gray-300 mb-3">
            Favoritos ({favorites.length})
          </h4>
          {favorites.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <Star className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No hay favoritos aún</p>
            </div>
          ) : (
            <div className="space-y-2">
              {favorites.map((tab) => (
                <div
                  key={tab}
                  className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg border border-gray-600"
                >
                  <div className="flex items-center gap-2">
                    <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                    <span className="text-white capitalize">{tab}</span>
                  </div>
                  <button
                    onClick={() => handleRemoveFavorite(tab)}
                    className="p-1 text-gray-400 hover:text-red-400 transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


