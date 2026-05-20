'use client';

import { useState } from 'react';
import { Users, Heart, MessageCircle, Share2, UserPlus } from 'lucide-react';
import toast from 'react-hot-toast';

interface SocialPost {
  id: string;
  user: string;
  avatar: string;
  content: string;
  trackName?: string;
  artistName?: string;
  likes: number;
  comments: number;
  shares: number;
  timestamp: string;
  liked: boolean;
}

export function MusicSocial() {
  const [posts] = useState<SocialPost[]>([
    {
      id: '1',
      user: 'MusicLover23',
      avatar: '🎵',
      content: 'Acabo de descubrir esta increíble canción!',
      trackName: 'Bohemian Rhapsody',
      artistName: 'Queen',
      likes: 42,
      comments: 8,
      shares: 5,
      timestamp: 'Hace 2 horas',
      liked: false,
    },
    {
      id: '2',
      user: 'BeatHunter',
      avatar: '🎶',
      content: 'El análisis de esta canción es impresionante',
      trackName: 'Stairway to Heaven',
      artistName: 'Led Zeppelin',
      likes: 38,
      comments: 12,
      shares: 3,
      timestamp: 'Hace 5 horas',
      liked: true,
    },
    {
      id: '3',
      user: 'SoundExplorer',
      avatar: '🎼',
      content: 'Compartiendo mi playlist favorita del mes',
      likes: 25,
      comments: 6,
      shares: 9,
      timestamp: 'Hace 1 día',
      liked: false,
    },
  ]);

  const handleLike = (postId: string) => {
    toast.success('Like agregado');
  };

  const handleComment = (postId: string) => {
    toast.info('Abrir comentarios');
  };

  const handleShare = (postId: string) => {
    toast.success('Compartido');
  };

  const handleFollow = (userId: string) => {
    toast.success('Usuario seguido');
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Users className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Social</h3>
      </div>

      <div className="space-y-4">
        {posts.map((post) => (
          <div key={post.id} className="p-4 bg-white/5 rounded-lg border border-white/10">
            <div className="flex items-start gap-3 mb-3">
              <div className="text-2xl">{post.avatar}</div>
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <span className="text-white font-semibold">{post.user}</span>
                    <button
                      onClick={() => handleFollow(post.user)}
                      className="px-2 py-1 text-xs bg-purple-600 hover:bg-purple-700 text-white rounded transition-colors flex items-center gap-1"
                    >
                      <UserPlus className="w-3 h-3" />
                      Seguir
                    </button>
                  </div>
                  <span className="text-xs text-gray-400">{post.timestamp}</span>
                </div>
                <p className="text-gray-300 text-sm mb-2">{post.content}</p>
                {post.trackName && (
                  <div className="p-2 bg-purple-500/20 rounded border border-purple-500/30">
                    <p className="text-white font-medium text-sm">{post.trackName}</p>
                    <p className="text-gray-400 text-xs">{post.artistName}</p>
                  </div>
                )}
              </div>
            </div>

            <div className="flex items-center gap-4 pt-3 border-t border-white/10">
              <button
                onClick={() => handleLike(post.id)}
                className={`flex items-center gap-2 text-sm transition-colors ${
                  post.liked ? 'text-red-400' : 'text-gray-400 hover:text-red-400'
                }`}
              >
                <Heart className={`w-4 h-4 ${post.liked ? 'fill-current' : ''}`} />
                {post.likes}
              </button>
              <button
                onClick={() => handleComment(post.id)}
                className="flex items-center gap-2 text-sm text-gray-400 hover:text-purple-400 transition-colors"
              >
                <MessageCircle className="w-4 h-4" />
                {post.comments}
              </button>
              <button
                onClick={() => handleShare(post.id)}
                className="flex items-center gap-2 text-sm text-gray-400 hover:text-purple-400 transition-colors"
              >
                <Share2 className="w-4 h-4" />
                {post.shares}
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}


