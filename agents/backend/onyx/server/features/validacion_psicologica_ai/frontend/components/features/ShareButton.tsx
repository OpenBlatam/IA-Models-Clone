/**
 * Share button component for sharing validations
 */

'use client';

import React, { useState } from 'react';
import { Button, Dropdown } from '@/components/ui';
import { Share2, Link2, Mail, Twitter, Facebook } from 'lucide-react';
import toast from 'react-hot-toast';

export interface ShareButtonProps {
  validationId: string;
  title?: string;
}

const ShareButton: React.FC<ShareButtonProps> = ({ validationId, title = 'Validación' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const shareUrl = typeof window !== 'undefined' 
    ? `${window.location.origin}/validations/${validationId}`
    : '';

  const handleCopyLink = async () => {
    if (!shareUrl) {
      return;
    }

    try {
      await navigator.clipboard.writeText(shareUrl);
      toast.success('Enlace copiado al portapapeles');
      setIsOpen(false);
    } catch (error) {
      toast.error('Error al copiar enlace');
    }
  };

  const handleEmailShare = () => {
    const subject = encodeURIComponent(`Compartir ${title}`);
    const body = encodeURIComponent(`Mira esta validación: ${shareUrl}`);
    window.open(`mailto:?subject=${subject}&body=${body}`);
    setIsOpen(false);
  };

  const handleTwitterShare = () => {
    const text = encodeURIComponent(`Mira esta ${title}`);
    window.open(`https://twitter.com/intent/tweet?url=${encodeURIComponent(shareUrl)}&text=${text}`);
    setIsOpen(false);
  };

  const handleFacebookShare = () => {
    window.open(`https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareUrl)}`);
    setIsOpen(false);
  };

  const shareOptions = [
    {
      value: 'copy',
      label: 'Copiar enlace',
      icon: <Link2 className="h-4 w-4" />,
      onClick: handleCopyLink,
    },
    {
      value: 'email',
      label: 'Compartir por email',
      icon: <Mail className="h-4 w-4" />,
      onClick: handleEmailShare,
    },
    {
      value: 'twitter',
      label: 'Compartir en Twitter',
      icon: <Twitter className="h-4 w-4" />,
      onClick: handleTwitterShare,
    },
    {
      value: 'facebook',
      label: 'Compartir en Facebook',
      icon: <Facebook className="h-4 w-4" />,
      onClick: handleFacebookShare,
    },
  ];

  const handleToggle = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className="relative">
      <Button
        variant="outline"
        size="sm"
        onClick={handleToggle}
        aria-label="Compartir validación"
        aria-haspopup="true"
        aria-expanded={isOpen}
        tabIndex={0}
      >
        <Share2 className="h-4 w-4 mr-2" aria-hidden="true" />
        Compartir
      </Button>
      {isOpen && (
        <div
          className="absolute right-0 mt-2 w-56 bg-background border border-input rounded-md shadow-lg z-50"
          role="menu"
          aria-label="Opciones de compartir"
        >
          {shareOptions.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={option.onClick}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  option.onClick();
                }
              }}
              className="w-full flex items-center gap-2 px-4 py-2 text-sm text-left hover:bg-accent hover:text-accent-foreground transition-colors focus-visible:outline-none focus-visible:bg-accent"
              role="menuitem"
              aria-label={option.label}
              tabIndex={0}
            >
              {option.icon}
              <span>{option.label}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export { ShareButton };




