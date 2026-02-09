'use client';

import { useEffect, useRef, useState } from 'react';
import { FiSave } from 'react-icons/fi';
import { motion, AnimatePresence } from 'framer-motion';

interface AutoSaveProps {
  data: any;
  onSave: (data: any) => void;
  interval?: number;
  storageKey?: string;
}

export default function AutoSave({
  data,
  onSave,
  interval = 30000, // 30 seconds
  storageKey = 'bul_autosave',
}: AutoSaveProps) {
  const [isSaving, setIsSaving] = useState(false);
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const dataRef = useRef(data);

  useEffect(() => {
    dataRef.current = data;
  }, [data]);

  useEffect(() => {
    // Load from localStorage on mount
    const saved = localStorage.getItem(storageKey);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        onSave(parsed);
        setLastSaved(new Date(parsed.timestamp || Date.now()));
      } catch (error) {
        console.error('Error loading autosave:', error);
      }
    }

    // Auto-save interval
    intervalRef.current = setInterval(() => {
      setIsSaving(true);
      const dataToSave = {
        ...dataRef.current,
        timestamp: Date.now(),
      };
      localStorage.setItem(storageKey, JSON.stringify(dataToSave));
      setLastSaved(new Date());
      setTimeout(() => setIsSaving(false), 500);
    }, interval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [interval, storageKey, onSave]);

  // Save on unmount
  useEffect(() => {
    return () => {
      const dataToSave = {
        ...dataRef.current,
        timestamp: Date.now(),
      };
      localStorage.setItem(storageKey, JSON.stringify(dataToSave));
    };
  }, [storageKey]);

  return (
    <AnimatePresence>
      {isSaving && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 20 }}
          className="fixed bottom-4 right-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-3 flex items-center gap-2 z-50"
        >
          <FiSave size={16} className="text-primary-600 animate-pulse" />
          <span className="text-sm text-gray-700 dark:text-gray-300">Guardando...</span>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

