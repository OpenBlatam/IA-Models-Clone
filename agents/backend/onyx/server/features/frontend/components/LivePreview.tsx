'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { FiEye, FiX } from 'react-icons/fi';

interface LivePreviewProps {
  content: string;
  isVisible: boolean;
  onClose: () => void;
}

export default function LivePreview({ content, isVisible, onClose }: LivePreviewProps) {
  if (!isVisible) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, x: 300 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: 300 }}
        className="fixed right-0 top-0 h-full w-1/2 bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 shadow-xl z-40 overflow-y-auto"
      >
        <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 flex items-center justify-between z-10">
          <div className="flex items-center gap-2">
            <FiEye size={20} className="text-primary-600" />
            <h3 className="font-semibold text-gray-900 dark:text-white">Vista Previa</h3>
          </div>
          <button onClick={onClose} className="btn-icon">
            <FiX size={20} />
          </button>
        </div>
        <div className="p-6">
          <div className="prose prose-sm max-w-none dark:prose-invert">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {content || '*Escribe algo para ver la vista previa...*'}
            </ReactMarkdown>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}


