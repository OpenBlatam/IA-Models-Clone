/**
 * Post Card Component
 * Reusable component for displaying a single post
 */

'use client';

import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { formatDate, getPlatformIcon, getStatusColor } from '@/lib/utils';
import { Post } from '@/types';
import { Edit, Trash2, Send } from 'lucide-react';

interface PostCardProps {
  post: Post;
  onEdit: (post: Post) => void;
  onDelete: (postId: string) => void;
  onPublish?: (postId: string) => void;
  isLoading?: boolean;
}

/**
 * Post card component with actions
 */
export const PostCard = ({ post, onEdit, onDelete, onPublish, isLoading }: PostCardProps) => {
  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2 flex-wrap">
              <Badge variant={post.status === 'published' ? 'success' : post.status === 'scheduled' ? 'warning' : 'danger'}>
                {post.status}
              </Badge>
              {post.scheduled_time && (
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {formatDate(post.scheduled_time)}
                </span>
              )}
            </div>
            
            <p className="text-gray-900 dark:text-gray-100 mb-3 break-words">{post.content}</p>
            
            <div className="flex items-center gap-2 flex-wrap">
              {post.platforms.map((platform) => (
                <span
                  key={platform}
                  className="inline-flex items-center gap-1 px-2 py-1 text-xs bg-gray-100 dark:bg-gray-800 rounded"
                  aria-label={`Plataforma: ${platform}`}
                >
                  {getPlatformIcon(platform)} {platform}
                </span>
              ))}
              
              {post.tags && post.tags.length > 0 && (
                <div className="flex items-center gap-1 flex-wrap">
                  {post.tags.map((tag) => (
                    <Badge key={tag} variant="info" size="sm">
                      #{tag}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </div>
          
          <div className="flex items-center gap-2 flex-shrink-0">
            {post.status === 'scheduled' && onPublish && (
              <Button
                size="sm"
                variant="primary"
                onClick={() => onPublish(post.post_id)}
                disabled={isLoading}
                aria-label="Publicar ahora"
                title="Publicar ahora"
              >
                <Send className="h-4 w-4" />
              </Button>
            )}
            <Button
              size="sm"
              variant="ghost"
              onClick={() => onEdit(post)}
              disabled={isLoading}
              aria-label="Editar post"
              title="Editar post"
            >
              <Edit className="h-4 w-4" />
            </Button>
            <Button
              size="sm"
              variant="danger"
              onClick={() => onDelete(post.post_id)}
              disabled={isLoading}
              aria-label="Eliminar post"
              title="Eliminar post"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};


