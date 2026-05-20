'use client';

import { useState } from 'react';
import { Share2, Copy, Mail, Twitter, Facebook, Link2 } from 'lucide-react';
import toast from 'react-hot-toast';

interface ShareMenuProps {
  url: string;
  title: string;
  description?: string;
}

export function ShareMenu({ url, title, description }: ShareMenuProps) {
  const [isOpen, setIsOpen] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(url);
    toast.success('Enlace copiado');
    setIsOpen(false);
  };

  const handleShare = async (platform: string) => {
    const shareText = `${title}${description ? ` - ${description}` : ''}`;
    const shareUrl = encodeURIComponent(url);

    let shareLink = '';

    switch (platform) {
      case 'twitter':
        shareLink = `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${shareUrl}`;
        break;
      case 'facebook':
        shareLink = `https://www.facebook.com/sharer/sharer.php?u=${shareUrl}`;
        break;
      case 'email':
        shareLink = `mailto:?subject=${encodeURIComponent(title)}&body=${encodeURIComponent(`${shareText}\n\n${url}`)}`;
        break;
      default:
        return;
    }

    window.open(shareLink, '_blank');
    setIsOpen(false);
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
      >
        <Share2 className="w-5 h-5" />
        <span>Compartir</span>
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute right-0 mt-2 w-48 bg-gray-800 rounded-lg shadow-lg border border-white/20 z-20">
            <div className="p-2">
              <button
                onClick={handleCopy}
                className="w-full flex items-center gap-3 px-3 py-2 text-white hover:bg-white/10 rounded-lg transition-colors"
              >
                <Copy className="w-4 h-4" />
                <span>Copiar enlace</span>
              </button>
              <button
                onClick={() => handleShare('twitter')}
                className="w-full flex items-center gap-3 px-3 py-2 text-white hover:bg-white/10 rounded-lg transition-colors"
              >
                <Twitter className="w-4 h-4" />
                <span>Twitter</span>
              </button>
              <button
                onClick={() => handleShare('facebook')}
                className="w-full flex items-center gap-3 px-3 py-2 text-white hover:bg-white/10 rounded-lg transition-colors"
              >
                <Facebook className="w-4 h-4" />
                <span>Facebook</span>
              </button>
              <button
                onClick={() => handleShare('email')}
                className="w-full flex items-center gap-3 px-3 py-2 text-white hover:bg-white/10 rounded-lg transition-colors"
              >
                <Mail className="w-4 h-4" />
                <span>Email</span>
              </button>
              {navigator.share && (
                <button
                  onClick={async () => {
                    try {
                      await navigator.share({
                        title,
                        text: description,
                        url,
                      });
                      setIsOpen(false);
                    } catch (err) {
                      // Usuario canceló
                    }
                  }}
                  className="w-full flex items-center gap-3 px-3 py-2 text-white hover:bg-white/10 rounded-lg transition-colors"
                >
                  <Link2 className="w-4 h-4" />
                  <span>Compartir nativo</span>
                </button>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}


