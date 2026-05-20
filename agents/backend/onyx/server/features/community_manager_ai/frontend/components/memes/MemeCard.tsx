/**
 * Meme Card Component
 * Reusable component for displaying a single meme
 */

'use client';

import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { LazyImage } from '@/components/ui/LazyImage';
import { Meme } from '@/types';
import { Trash2 } from 'lucide-react';

interface MemeCardProps {
  meme: Meme;
  onDelete: (memeId: string) => void;
  isLoading?: boolean;
}

/**
 * Meme card component with image and actions
 */
export const MemeCard = ({ meme, onDelete, isLoading }: MemeCardProps) => {
  return (
    <Card className="overflow-hidden hover:shadow-lg transition-shadow">
      <div className="relative aspect-square bg-gray-100 dark:bg-gray-800">
        <LazyImage
          src={meme.image_path}
          alt={meme.caption || 'Meme'}
          className="h-full w-full object-cover"
          loading="lazy"
        />
      </div>
      <CardContent className="p-4">
        {meme.caption && (
          <p className="mb-2 text-sm text-gray-900 dark:text-gray-100 line-clamp-2">
            {meme.caption}
          </p>
        )}
        
        {meme.tags && meme.tags.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-1">
            {meme.tags.map((tag) => (
              <Badge key={tag} variant="info" size="sm">
                #{tag}
              </Badge>
            ))}
          </div>
        )}
        
        {meme.category && (
          <p className="mb-2 text-xs text-gray-500 dark:text-gray-400">
            {meme.category}
          </p>
        )}
        
        <Button
          size="sm"
          variant="danger"
          className="w-full"
          onClick={() => onDelete(meme.meme_id)}
          disabled={isLoading}
          aria-label="Eliminar meme"
        >
          <Trash2 className="mr-2 h-4 w-4" />
          Eliminar
        </Button>
      </CardContent>
    </Card>
  );
};


