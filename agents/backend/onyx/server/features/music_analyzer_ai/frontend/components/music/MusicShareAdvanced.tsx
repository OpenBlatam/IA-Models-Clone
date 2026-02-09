'use client';

import { useState } from 'react';
import { Share2, Copy, Check, QrCode, Link2 } from 'lucide-react';
import toast from 'react-hot-toast';

interface MusicShareAdvancedProps {
  url: string;
  title: string;
  description?: string;
}

export function MusicShareAdvanced({ url, title, description }: MusicShareAdvancedProps) {
  const [copied, setCopied] = useState(false);
  const [showQR, setShowQR] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(url);
    setCopied(true);
    toast.success('Enlace copiado');
    setTimeout(() => setCopied(false), 2000);
  };

  const handleGenerateQR = () => {
    setShowQR(!showQR);
    // En producción, esto generaría un código QR real
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Share2 className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Compartir Avanzado</h3>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm text-gray-400 mb-2">Enlace</label>
          <div className="flex gap-2">
            <input
              type="text"
              value={url}
              readOnly
              className="flex-1 px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white text-sm"
            />
            <button
              onClick={handleCopy}
              className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors flex items-center gap-2"
            >
              {copied ? (
                <>
                  <Check className="w-4 h-4" />
                  Copiado
                </>
              ) : (
                <>
                  <Copy className="w-4 h-4" />
                  Copiar
                </>
              )}
            </button>
          </div>
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-2">Título</label>
          <input
            type="text"
            value={title}
            readOnly
            className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white text-sm"
          />
        </div>

        {description && (
          <div>
            <label className="block text-sm text-gray-400 mb-2">Descripción</label>
            <textarea
              value={description}
              readOnly
              rows={3}
              className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white text-sm"
            />
          </div>
        )}

        <button
          onClick={handleGenerateQR}
          className="w-full px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
        >
          <QrCode className="w-4 h-4" />
          {showQR ? 'Ocultar' : 'Generar'} Código QR
        </button>

        {showQR && (
          <div className="p-4 bg-white/5 rounded-lg border border-white/10 text-center">
            <div className="w-48 h-48 mx-auto bg-white rounded-lg flex items-center justify-center mb-2">
              <QrCode className="w-32 h-32 text-gray-400" />
            </div>
            <p className="text-xs text-gray-400">Código QR (simulado)</p>
          </div>
        )}
      </div>
    </div>
  );
}


