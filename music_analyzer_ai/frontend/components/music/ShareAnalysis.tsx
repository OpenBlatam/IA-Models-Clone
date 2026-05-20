'use client';

import { useState } from 'react';
import { Share2, Copy, Check, Link as LinkIcon } from 'lucide-react';
import toast from 'react-hot-toast';

interface ShareAnalysisProps {
  trackId: string;
  trackName: string;
  analysisId?: string;
}

export function ShareAnalysis({ trackId, trackName, analysisId }: ShareAnalysisProps) {
  const [copied, setCopied] = useState(false);

  const shareUrl = typeof window !== 'undefined'
    ? `${window.location.origin}/music/analysis/${analysisId || trackId}`
    : '';

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      toast.success('Enlace copiado al portapapeles');
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      toast.error('Error al copiar enlace');
    }
  };

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: `Análisis de ${trackName}`,
          text: `Mira el análisis musical de ${trackName}`,
          url: shareUrl,
        });
        toast.success('Compartido exitosamente');
      } catch (error: any) {
        if (error.name !== 'AbortError') {
          toast.error('Error al compartir');
        }
      }
    } else {
      handleCopy();
    }
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
      <div className="flex items-center gap-2 mb-3">
        <Share2 className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Compartir Análisis</h3>
      </div>

      <div className="flex gap-2">
        <button
          onClick={handleShare}
          className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
        >
          <Share2 className="w-4 h-4" />
          Compartir
        </button>
        <button
          onClick={handleCopy}
          className="flex items-center justify-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
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

      <div className="mt-3 p-2 bg-white/5 rounded-lg">
        <div className="flex items-center gap-2">
          <LinkIcon className="w-4 h-4 text-gray-400" />
          <input
            type="text"
            value={shareUrl}
            readOnly
            className="flex-1 bg-transparent text-sm text-gray-300 focus:outline-none"
          />
        </div>
      </div>
    </div>
  );
}

