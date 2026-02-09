'use client';

import { useState } from 'react';
import { Share2, Copy, Check, Twitter, Facebook, Mail } from 'lucide-react';
import toast from 'react-hot-toast';

interface MusicShareProps {
  trackId: string;
  trackName: string;
  artists: string[];
  analysisId?: string;
}

export function MusicShare({ trackId, trackName, artists, analysisId }: MusicShareProps) {
  const [copied, setCopied] = useState(false);

  const shareUrl = `${window.location.origin}/music/analysis/${trackId}${analysisId ? `?analysis=${analysisId}` : ''}`;
  const shareText = `${trackName} - ${artists.join(', ')}`;

  const handleCopy = () => {
    navigator.clipboard.writeText(shareUrl);
    setCopied(true);
    toast.success('Enlace copiado');
    setTimeout(() => setCopied(false), 2000);
  };

  const handleShare = (platform: string) => {
    const encodedUrl = encodeURIComponent(shareUrl);
    const encodedText = encodeURIComponent(shareText);

    let shareLink = '';

    switch (platform) {
      case 'twitter':
        shareLink = `https://twitter.com/intent/tweet?text=${encodedText}&url=${encodedUrl}`;
        break;
      case 'facebook':
        shareLink = `https://www.facebook.com/sharer/sharer.php?u=${encodedUrl}`;
        break;
      case 'email':
        shareLink = `mailto:?subject=${encodeURIComponent(trackName)}&body=${encodedText}%0A%0A${encodedUrl}`;
        break;
      default:
        return;
    }

    window.open(shareLink, '_blank');
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Share2 className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Compartir</h3>
      </div>

      <div className="space-y-3">
        <div className="flex gap-2">
          <input
            type="text"
            value={shareUrl}
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

        <div className="flex gap-2">
          <button
            onClick={() => handleShare('twitter')}
            className="flex-1 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <Twitter className="w-4 h-4" />
            Twitter
          </button>
          <button
            onClick={() => handleShare('facebook')}
            className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <Facebook className="w-4 h-4" />
            Facebook
          </button>
          <button
            onClick={() => handleShare('email')}
            className="flex-1 px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <Mail className="w-4 h-4" />
            Email
          </button>
        </div>

        {navigator.share && (
          <button
            onClick={async () => {
              try {
                await navigator.share({
                  title: trackName,
                  text: shareText,
                  url: shareUrl,
                });
              } catch (err) {
                // Usuario canceló
              }
            }}
            className="w-full px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <Share2 className="w-4 h-4" />
            Compartir nativo
          </button>
        )}
      </div>
    </div>
  );
}


