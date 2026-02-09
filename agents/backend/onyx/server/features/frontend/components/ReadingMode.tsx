'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiBook, FiX, FiSettings, FiMinus, FiPlus } from 'react-icons/fi';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ReadingModeProps {
  content: string;
  title: string;
  onClose: () => void;
}

export default function ReadingMode({ content, title, onClose }: ReadingModeProps) {
  const [fontSize, setFontSize] = useState(16);
  const [fontFamily, setFontFamily] = useState('Inter');
  const [lineHeight, setLineHeight] = useState(1.6);
  const [maxWidth, setMaxWidth] = useState(800);
  const [showSettings, setShowSettings] = useState(false);

  const fontFamilies = [
    { name: 'Inter', value: 'Inter, sans-serif' },
    { name: 'Georgia', value: 'Georgia, serif' },
    { name: 'Monaco', value: 'Monaco, monospace' },
    { name: 'Arial', value: 'Arial, sans-serif' },
  ];

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-gray-50 dark:bg-gray-900 z-50 overflow-y-auto"
      >
        {/* Header */}
        <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 flex items-center justify-between z-10">
          <div className="flex items-center gap-3">
            <FiBook size={20} className="text-primary-600" />
            <h2 className="font-semibold text-gray-900 dark:text-white">{title}</h2>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="btn-icon"
              title="Configuración"
            >
              <FiSettings size={18} />
            </button>
            <button onClick={onClose} className="btn-icon">
              <FiX size={18} />
            </button>
          </div>
        </div>

        {/* Settings Panel */}
        <AnimatePresence>
          {showSettings && (
            <motion.div
              initial={{ y: -100, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: -100, opacity: 0 }}
              className="sticky top-[73px] bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 z-10"
            >
              <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-4">
                <div>
                  <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Tamaño de Fuente
                  </label>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setFontSize(Math.max(12, fontSize - 2))}
                      className="btn-icon"
                    >
                      <FiMinus size={14} />
                    </button>
                    <span className="text-sm font-medium w-12 text-center">{fontSize}px</span>
                    <button
                      onClick={() => setFontSize(Math.min(24, fontSize + 2))}
                      className="btn-icon"
                    >
                      <FiPlus size={14} />
                    </button>
                  </div>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Fuente
                  </label>
                  <select
                    value={fontFamily}
                    onChange={(e) => setFontFamily(e.target.value)}
                    className="select text-sm"
                  >
                    {fontFamilies.map((font) => (
                      <option key={font.value} value={font.name}>
                        {font.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Interlineado
                  </label>
                  <input
                    type="range"
                    min="1.2"
                    max="2.4"
                    step="0.1"
                    value={lineHeight}
                    onChange={(e) => setLineHeight(parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <span className="text-xs text-gray-500">{lineHeight.toFixed(1)}</span>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Ancho Máximo
                  </label>
                  <input
                    type="range"
                    min="600"
                    max="1200"
                    step="50"
                    value={maxWidth}
                    onChange={(e) => setMaxWidth(parseInt(e.target.value))}
                    className="w-full"
                  />
                  <span className="text-xs text-gray-500">{maxWidth}px</span>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Content */}
        <div className="max-w-7xl mx-auto p-8">
          <div
            style={{
              maxWidth: `${maxWidth}px`,
              margin: '0 auto',
              fontSize: `${fontSize}px`,
              fontFamily: fontFamilies.find((f) => f.name === fontFamily)?.value || 'Inter',
              lineHeight: lineHeight,
            }}
            className="prose prose-lg max-w-none dark:prose-invert"
          >
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}


