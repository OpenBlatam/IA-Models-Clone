'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { MessageSquare, Send, Trash2, Edit2 } from 'lucide-react';
import toast from 'react-hot-toast';

interface CommentSectionProps {
  resourceId: string;
  resourceType: 'track' | 'analysis' | 'playlist';
}

export function CommentSection({ resourceId, resourceType }: CommentSectionProps) {
  const [newComment, setNewComment] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState('');
  const queryClient = useQueryClient();

  const { data: commentsData } = useQuery({
    queryKey: ['comments', resourceId, resourceType],
    queryFn: () => musicApiService.getComments?.(resourceId, resourceType) || Promise.resolve({ comments: [] }),
  });

  const addCommentMutation = useMutation({
    mutationFn: (content: string) =>
      musicApiService.addComment?.(resourceId, resourceType, content) || Promise.resolve({}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['comments', resourceId, resourceType] });
      toast.success('Comentario agregado');
      setNewComment('');
    },
    onError: () => {
      toast.error('Error al agregar comentario');
    },
  });

  const updateCommentMutation = useMutation({
    mutationFn: ({ commentId, content }: { commentId: string; content: string }) =>
      musicApiService.updateComment?.(commentId, content) || Promise.resolve({}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['comments', resourceId, resourceType] });
      toast.success('Comentario actualizado');
      setEditingId(null);
      setEditText('');
    },
    onError: () => {
      toast.error('Error al actualizar comentario');
    },
  });

  const deleteCommentMutation = useMutation({
    mutationFn: (commentId: string) =>
      musicApiService.deleteComment?.(commentId) || Promise.resolve({}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['comments', resourceId, resourceType] });
      toast.success('Comentario eliminado');
    },
    onError: () => {
      toast.error('Error al eliminar comentario');
    },
  });

  const comments = commentsData?.comments || [];

  const handleAddComment = () => {
    if (!newComment.trim()) {
      toast.error('Escribe un comentario');
      return;
    }
    addCommentMutation.mutate(newComment.trim());
  };

  const handleStartEdit = (comment: any) => {
    setEditingId(comment.id);
    setEditText(comment.content);
  };

  const handleUpdateComment = (commentId: string) => {
    if (!editText.trim()) {
      toast.error('El comentario no puede estar vacío');
      return;
    }
    updateCommentMutation.mutate({ commentId, content: editText.trim() });
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <MessageSquare className="w-6 h-6 text-purple-300" />
        <h2 className="text-2xl font-semibold text-white">Comentarios</h2>
        <span className="text-sm text-gray-400">({comments.length})</span>
      </div>

      <div className="mb-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={newComment}
            onChange={(e) => setNewComment(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleAddComment()}
            placeholder="Escribe un comentario..."
            className="flex-1 px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
          />
          <button
            onClick={handleAddComment}
            disabled={addCommentMutation.isPending}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
          >
            <Send className="w-4 h-4" />
            Enviar
          </button>
        </div>
      </div>

      <div className="space-y-3 max-h-96 overflow-y-auto">
        {comments.length === 0 ? (
          <div className="text-center py-8">
            <MessageSquare className="w-12 h-12 text-gray-500 mx-auto mb-2" />
            <p className="text-gray-400">No hay comentarios aún</p>
          </div>
        ) : (
          comments.map((comment: any) => (
            <div
              key={comment.id}
              className="p-4 bg-white/5 rounded-lg border border-white/10"
            >
              {editingId === comment.id ? (
                <div>
                  <input
                    type="text"
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                    className="w-full px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white mb-2"
                  />
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleUpdateComment(comment.id)}
                      className="px-3 py-1 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-sm"
                    >
                      Guardar
                    </button>
                    <button
                      onClick={() => {
                        setEditingId(null);
                        setEditText('');
                      }}
                      className="px-3 py-1 bg-white/10 hover:bg-white/20 text-white rounded-lg text-sm"
                    >
                      Cancelar
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <p className="text-white mb-2">{comment.content}</p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-400">
                      {comment.user_name || 'Usuario'} •{' '}
                      {comment.created_at
                        ? new Date(comment.created_at).toLocaleDateString()
                        : 'Fecha desconocida'}
                    </span>
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleStartEdit(comment)}
                        className="p-1 text-gray-400 hover:text-white transition-colors"
                      >
                        <Edit2 className="w-3 h-3" />
                      </button>
                      <button
                        onClick={() => deleteCommentMutation.mutate(comment.id)}
                        className="p-1 text-gray-400 hover:text-red-300 transition-colors"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                </>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}


