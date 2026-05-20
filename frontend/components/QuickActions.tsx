'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { FiZap, FiFileText, FiDownload, FiShare2, FiStar } from 'react-icons/fi';

interface QuickAction {
  id: string;
  label: string;
  icon: React.ComponentType<{ size?: number; className?: string }>;
  action: () => void;
  color: string;
}

interface QuickActionsProps {
  onGenerate?: () => void;
  onViewDocuments?: () => void;
  onViewFavorites?: () => void;
}

export default function QuickActions({ onGenerate, onViewDocuments, onViewFavorites }: QuickActionsProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const actions: QuickAction[] = [
    {
      id: 'generate',
      label: 'Generar',
      icon: FiZap,
      action: () => {
        onGenerate?.();
        setIsExpanded(false);
      },
      color: 'bg-blue-500',
    },
    {
      id: 'documents',
      label: 'Documentos',
      icon: FiFileText,
      action: () => {
        onViewDocuments?.();
        setIsExpanded(false);
      },
      color: 'bg-green-500',
    },
    {
      id: 'favorites',
      label: 'Favoritos',
      icon: FiStar,
      action: () => {
        onViewFavorites?.();
        setIsExpanded(false);
      },
      color: 'bg-yellow-500',
    },
  ];

  return (
    <div className="fixed bottom-6 right-6 z-40">
      <motion.div
        initial={false}
        animate={{ scale: isExpanded ? 1 : 0.8 }}
        className="relative"
      >
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="absolute bottom-16 right-0 mb-2 space-y-2"
          >
            {actions.map((action, index) => (
              <motion.button
                key={action.id}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                onClick={action.action}
                className={`${action.color} text-white p-4 rounded-full shadow-lg hover:shadow-xl transition-shadow flex items-center gap-3 group`}
                title={action.label}
              >
                <action.icon size={20} />
                <span className="text-sm font-medium opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                  {action.label}
                </span>
              </motion.button>
            ))}
          </motion.div>
        )}

        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="bg-primary-600 text-white p-4 rounded-full shadow-lg hover:shadow-xl transition-shadow"
          title="Acciones rápidas"
        >
          <FiZap size={24} className={isExpanded ? 'rotate-45' : ''} />
        </button>
      </motion.div>
    </div>
  );
}


