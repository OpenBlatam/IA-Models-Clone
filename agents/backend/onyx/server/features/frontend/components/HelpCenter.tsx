'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiHelpCircle, FiX, FiSearch, FiBook, FiVideo, FiMessageCircle } from 'react-icons/fi';

interface HelpArticle {
  id: string;
  title: string;
  content: string;
  category: string;
  tags: string[];
}

const helpArticles: HelpArticle[] = [
  {
    id: 'getting-started',
    title: 'Comenzar con BUL',
    content: 'BUL es una plataforma para generar documentos de negocio con IA. Comienza creando tu primer documento desde la vista "Generar".',
    category: 'Básico',
    tags: ['inicio', 'primeros pasos'],
  },
  {
    id: 'generating-documents',
    title: 'Generar Documentos',
    content: 'Usa el formulario de generación para crear documentos. Puedes usar plantillas, escribir manualmente, usar voz o arrastrar archivos.',
    category: 'Básico',
    tags: ['generar', 'documentos', 'plantillas'],
  },
  {
    id: 'keyboard-shortcuts',
    title: 'Atajos de Teclado',
    content: 'Presiona Ctrl+/ para ver todos los atajos disponibles. Usa Ctrl+K para búsqueda global, Ctrl+P para vista previa, etc.',
    category: 'Productividad',
    tags: ['atajos', 'teclado', 'productividad'],
  },
  {
    id: 'exporting',
    title: 'Exportar Documentos',
    content: 'Puedes exportar documentos en múltiples formatos: Markdown, Texto, HTML, PDF, JSON y XML. Usa el botón de exportación avanzada.',
    category: 'Funcionalidades',
    tags: ['exportar', 'formato', 'descargar'],
  },
  {
    id: 'favorites',
    title: 'Sistema de Favoritos',
    content: 'Marca documentos como favoritos para acceso rápido. Los favoritos se guardan localmente y persisten entre sesiones.',
    category: 'Organización',
    tags: ['favoritos', 'organización'],
  },
  {
    id: 'troubleshooting',
    title: 'Solución de Problemas',
    content: 'Si tienes problemas, verifica tu conexión a internet, revisa la consola del navegador o contacta al soporte.',
    category: 'Soporte',
    tags: ['problemas', 'soporte', 'ayuda'],
  },
];

interface HelpCenterProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function HelpCenter({ isOpen, onClose }: HelpCenterProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedArticle, setSelectedArticle] = useState<HelpArticle | null>(null);

  const categories = ['all', ...Array.from(new Set(helpArticles.map((a) => a.category)))];

  const filteredArticles = helpArticles.filter((article) => {
    const matchesSearch =
      article.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      article.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
      article.tags.some((tag) => tag.toLowerCase().includes(searchQuery.toLowerCase()));

    const matchesCategory = selectedCategory === 'all' || article.category === selectedCategory;

    return matchesSearch && matchesCategory;
  });

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-4xl w-full max-h-[90vh] flex flex-col"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="p-6 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <FiHelpCircle size={24} className="text-primary-600" />
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">Centro de Ayuda</h3>
            </div>
            <button onClick={onClose} className="btn-icon">
              <FiX size={20} />
            </button>
          </div>

          {selectedArticle ? (
            <div className="flex-1 overflow-y-auto p-6">
              <button
                onClick={() => setSelectedArticle(null)}
                className="text-primary-600 hover:text-primary-700 mb-4 flex items-center gap-2"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M19 12H5" />
                  <path d="M12 19l-7-7 7-7" />
                </svg>
                Volver
              </button>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                {selectedArticle.title}
              </h2>
              <div className="prose prose-sm max-w-none dark:prose-invert">
                <p className="text-gray-700 dark:text-gray-300">{selectedArticle.content}</p>
              </div>
            </div>
          ) : (
            <>
              <div className="p-6 border-b border-gray-200 dark:border-gray-700">
                <div className="relative mb-4">
                  <FiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Buscar ayuda..."
                    className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>
                <div className="flex gap-2 flex-wrap">
                  {categories.map((category) => (
                    <button
                      key={category}
                      onClick={() => setSelectedCategory(category)}
                      className={`px-3 py-1 rounded-full text-sm transition-colors ${
                        selectedCategory === category
                          ? 'bg-primary-600 text-white'
                          : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                      }`}
                    >
                      {category === 'all' ? 'Todas' : category}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex-1 overflow-y-auto p-6">
                {filteredArticles.length === 0 ? (
                  <div className="text-center py-12 text-gray-500 dark:text-gray-400">
                    <FiHelpCircle size={48} className="mx-auto mb-2 opacity-50" />
                    <p>No se encontraron artículos</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {filteredArticles.map((article) => (
                      <motion.button
                        key={article.id}
                        onClick={() => setSelectedArticle(article)}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="text-left p-4 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
                      >
                        <div className="flex items-start justify-between mb-2">
                          <h4 className="font-semibold text-gray-900 dark:text-white">
                            {article.title}
                          </h4>
                          <span className="text-xs text-primary-600 bg-primary-100 dark:bg-primary-900/30 px-2 py-1 rounded">
                            {article.category}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
                          {article.content}
                        </p>
                        <div className="flex gap-1 mt-2">
                          {article.tags.slice(0, 3).map((tag) => (
                            <span
                              key={tag}
                              className="text-xs text-gray-500 dark:text-gray-400 bg-gray-200 dark:bg-gray-600 px-2 py-0.5 rounded"
                            >
                              #{tag}
                            </span>
                          ))}
                        </div>
                      </motion.button>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}

          <div className="p-6 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
                <button className="flex items-center gap-2 hover:text-primary-600">
                  <FiVideo size={16} />
                  Videos
                </button>
                <button className="flex items-center gap-2 hover:text-primary-600">
                  <FiBook size={16} />
                  Documentación
                </button>
                <button className="flex items-center gap-2 hover:text-primary-600">
                  <FiMessageCircle size={16} />
                  Contactar Soporte
                </button>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}


