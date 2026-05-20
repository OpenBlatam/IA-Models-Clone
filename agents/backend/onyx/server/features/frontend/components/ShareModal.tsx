'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiCopy, FiX, FiMail, FiLink2, FiCheck } from 'react-icons/fi';
import { showToast } from '@/lib/toast';

interface ShareModalProps {
  taskId: string;
  documentTitle: string;
  onClose: () => void;
}

export default function ShareModal({ taskId, documentTitle, onClose }: ShareModalProps) {
  const [copied, setCopied] = useState(false);
  const [email, setEmail] = useState('');

  const shareUrl = `${window.location.origin}/document/${taskId}`;
  const shareText = `Mira este documento generado: ${documentTitle}`;

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      showToast('Enlace copiado al portapapeles', 'success');
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      showToast('Error al copiar enlace', 'error');
    }
  };

  const handleShareEmail = () => {
    const subject = encodeURIComponent(`Documento: ${documentTitle}`);
    const body = encodeURIComponent(`${shareText}\n\n${shareUrl}`);
    window.location.href = `mailto:${email}?subject=${subject}&body=${body}`;
  };

  const handleShareNative = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: documentTitle,
          text: shareText,
          url: shareUrl,
        });
        showToast('Documento compartido', 'success');
      } catch (error) {
        // User cancelled
      }
    }
  };

  return (
    <AnimatePresence>
      <div
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          onClick={(e) => e.stopPropagation()}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-md w-full p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white">Compartir Documento</h3>
            <button onClick={onClose} className="btn-icon">
              <FiX size={20} />
            </button>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Enlace de compartir
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={shareUrl}
                  readOnly
                  className="flex-1 input bg-gray-50 dark:bg-gray-700"
                />
                <button
                  onClick={handleCopyLink}
                  className="btn btn-secondary"
                  title="Copiar enlace"
                >
                  {copied ? <FiCheck size={18} /> : <FiCopy size={18} />}
                </button>
              </div>
            </div>

            {navigator.share && (
              <button
                onClick={handleShareNative}
                className="w-full btn btn-primary"
              >
                <FiLink2 size={18} />
                Compartir con...
              </button>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Compartir por email
              </label>
              <div className="flex gap-2">
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="email@ejemplo.com"
                  className="flex-1 input"
                />
                <button
                  onClick={handleShareEmail}
                  className="btn btn-secondary"
                  disabled={!email}
                >
                  <FiMail size={18} />
                </button>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}


