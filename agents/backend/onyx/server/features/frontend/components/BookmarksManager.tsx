'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiBookmark, FiX, FiExternalLink } from 'react-icons/fi';
import { format } from 'date-fns';

interface Bookmark {
  id: string;
  taskId: string;
  title: string;
  query: string;
  createdAt: Date;
  note?: string;
}

interface BookmarksManagerProps {
  taskId?: string;
  title?: string;
  query?: string;
}

export function useBookmarks() {
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);

  useEffect(() => {
    const stored = localStorage.getItem('bul_bookmarks');
    if (stored) {
      setBookmarks(JSON.parse(stored).map((b: any) => ({
        ...b,
        createdAt: new Date(b.createdAt),
      })));
    }
  }, []);

  const addBookmark = (bookmark: Omit<Bookmark, 'id' | 'createdAt'>) => {
    const newBookmark: Bookmark = {
      ...bookmark,
      id: `bookmark_${Date.now()}`,
      createdAt: new Date(),
    };
    const updated = [newBookmark, ...bookmarks];
    setBookmarks(updated);
    localStorage.setItem('bul_bookmarks', JSON.stringify(updated));
  };

  const removeBookmark = (id: string) => {
    const updated = bookmarks.filter((b) => b.id !== id);
    setBookmarks(updated);
    localStorage.setItem('bul_bookmarks', JSON.stringify(updated));
  };

  return { bookmarks, addBookmark, removeBookmark };
}

export default function BookmarksManager({ taskId, title, query }: BookmarksManagerProps) {
  const { bookmarks, addBookmark, removeBookmark } = useBookmarks();
  const [isOpen, setIsOpen] = useState(false);
  const [note, setNote] = useState('');

  const handleAdd = () => {
    if (!taskId || !title) return;
    addBookmark({ taskId, title, query: query || '', note });
    setNote('');
    setIsOpen(false);
  };

  const isBookmarked = taskId ? bookmarks.some((b) => b.taskId === taskId) : false;

  if (!taskId) {
    return (
      <div className="card">
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <FiBookmark size={20} />
          Marcadores
        </h3>
        {bookmarks.length === 0 ? (
          <p className="text-gray-500 dark:text-gray-400">No hay marcadores guardados</p>
        ) : (
          <div className="space-y-2">
            {bookmarks.map((bookmark) => (
              <div
                key={bookmark.id}
                className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg flex items-start justify-between"
              >
                <div className="flex-1">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-1">
                    {bookmark.title}
                  </h4>
                  {bookmark.note && (
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                      {bookmark.note}
                    </p>
                  )}
                  <p className="text-xs text-gray-500 dark:text-gray-500">
                    {format(bookmark.createdAt, 'PPp')}
                  </p>
                </div>
                <button
                  onClick={() => removeBookmark(bookmark.id)}
                  className="btn-icon text-red-600 ml-2"
                >
                  <FiX size={16} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className={`btn-icon ${isBookmarked ? 'text-yellow-600' : ''}`}
        title="Agregar marcador"
      >
        <FiBookmark size={18} fill={isBookmarked ? 'currentColor' : 'none'} />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
            onClick={() => setIsOpen(false)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-md w-full p-6"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 className="text-xl font-bold mb-4">Agregar Marcador</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Nota (opcional)</label>
                  <textarea
                    value={note}
                    onChange={(e) => setNote(e.target.value)}
                    className="input w-full"
                    rows={3}
                    placeholder="Agrega una nota para este marcador..."
                  />
                </div>
                <div className="flex gap-2">
                  <button onClick={handleAdd} className="btn btn-primary flex-1">
                    Guardar
                  </button>
                  <button onClick={() => setIsOpen(false)} className="btn btn-secondary">
                    Cancelar
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

