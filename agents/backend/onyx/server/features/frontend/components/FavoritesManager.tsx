'use client';

import { useState, useEffect } from 'react';
import { FiStar, FiTrash2 } from 'react-icons/fi';
import { motion, AnimatePresence } from 'framer-motion';
import { showToast } from '@/lib/toast';

export interface Favorite {
  id: string;
  taskId: string;
  title: string;
  query: string;
  createdAt: string;
}

export default function useFavoritesManager() {
  const [favorites, setFavorites] = useState<Favorite[]>([]);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem('bul_favorites');
    if (stored) {
      setFavorites(JSON.parse(stored));
    }
  }, []);

  const saveFavorites = (newFavorites: Favorite[]) => {
    localStorage.setItem('bul_favorites', JSON.stringify(newFavorites));
    setFavorites(newFavorites);
  };

  const addFavorite = (taskId: string, title: string, query: string) => {
    const newFavorite: Favorite = {
      id: Date.now().toString(),
      taskId,
      title,
      query,
      createdAt: new Date().toISOString(),
    };
    const updated = [...favorites, newFavorite];
    saveFavorites(updated);
    showToast('Agregado a favoritos', 'success');
  };

  const removeFavorite = (id: string) => {
    const updated = favorites.filter((f) => f.id !== id);
    saveFavorites(updated);
    showToast('Eliminado de favoritos', 'success');
  };

  const isFavorite = (taskId: string) => {
    return favorites.some((f) => f.taskId === taskId);
  };

  return {
    favorites,
    isFavorite,
    addFavorite,
    removeFavorite,
    isOpen,
    setIsOpen,
  };
}

export function FavoriteButton({ taskId, title, query }: { taskId: string; title: string; query: string }) {
  const [favorited, setFavorited] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem('bul_favorites');
    if (stored) {
      const favorites: Favorite[] = JSON.parse(stored);
      setFavorited(favorites.some((f) => f.taskId === taskId));
    }
  }, [taskId]);

  const handleToggle = () => {
    const stored = localStorage.getItem('bul_favorites');
    const favorites: Favorite[] = stored ? JSON.parse(stored) : [];
    
    if (favorited) {
      const updated = favorites.filter((f) => f.taskId !== taskId);
      localStorage.setItem('bul_favorites', JSON.stringify(updated));
      setFavorited(false);
      showToast('Eliminado de favoritos', 'success');
    } else {
      const newFavorite: Favorite = {
        id: Date.now().toString(),
        taskId,
        title,
        query,
        createdAt: new Date().toISOString(),
      };
      const updated = [...favorites, newFavorite];
      localStorage.setItem('bul_favorites', JSON.stringify(updated));
      setFavorited(true);
      showToast('Agregado a favoritos', 'success');
    }
  };

  return (
    <button
      onClick={handleToggle}
      className={`btn-icon ${favorited ? 'text-yellow-500' : 'text-gray-400'}`}
      title={favorited ? 'Quitar de favoritos' : 'Agregar a favoritos'}
    >
      <FiStar size={18} fill={favorited ? 'currentColor' : 'none'} />
    </button>
  );
}

