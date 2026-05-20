'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FiStar, FiTrash2, FiEye } from 'react-icons/fi';
import type { Favorite } from '@/components/FavoritesManager';
import DocumentModal from '@/components/DocumentModal';
import { showToast } from '@/lib/toast';
import { format } from 'date-fns';

export default function FavoritesView() {
  const [favorites, setFavorites] = useState<Favorite[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);

  useEffect(() => {
    const stored = localStorage.getItem('bul_favorites');
    if (stored) {
      setFavorites(JSON.parse(stored));
    }
  }, []);

  const handleRemove = (id: string) => {
    const updated = favorites.filter((f) => f.id !== id);
    localStorage.setItem('bul_favorites', JSON.stringify(updated));
    setFavorites(updated);
    showToast('Eliminado de favoritos', 'success');
  };

  const handleView = (taskId: string) => {
    setSelectedTaskId(taskId);
  };

  return (
    <div>
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-6"
      >
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Favoritos</h2>
        <p className="text-gray-600 dark:text-gray-400">Tus documentos guardados</p>
      </motion.div>

      {favorites.length === 0 ? (
        <div className="card text-center py-12">
          <FiStar className="mx-auto h-16 w-16 text-gray-400 mb-4" />
          <p className="text-gray-600 dark:text-gray-400">No tienes favoritos aún</p>
          <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">
            Agrega documentos a favoritos para acceder rápidamente
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {favorites.map((favorite, index) => (
            <motion.div
              key={favorite.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className="card hover:shadow-md transition-shadow"
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-1 line-clamp-2">
                    {favorite.title}
                  </h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {format(new Date(favorite.createdAt), "PPp")}
                  </p>
                </div>
                <FiStar className="text-yellow-500 flex-shrink-0 ml-2" size={20} fill="currentColor" />
              </div>
              
              <p className="text-sm text-gray-600 dark:text-gray-300 mb-4 line-clamp-3">
                {favorite.query}
              </p>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleView(favorite.taskId)}
                  className="btn btn-primary flex-1 text-sm"
                >
                  <FiEye size={16} />
                  Ver
                </button>
                <button
                  onClick={() => handleRemove(favorite.id)}
                  className="btn btn-secondary text-sm"
                  title="Eliminar de favoritos"
                >
                  <FiTrash2 size={16} />
                </button>
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {selectedTaskId && (
        <DocumentModal taskId={selectedTaskId} onClose={() => setSelectedTaskId(null)} />
      )}
    </div>
  );
}


