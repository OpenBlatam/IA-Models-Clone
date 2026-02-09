'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FiMessageSquare, FiSend, FiTrash2 } from 'react-icons/fi';
import { format } from 'date-fns';

interface Comment {
  id: string;
  text: string;
  author: string;
  timestamp: Date;
}

interface CommentsPanelProps {
  taskId: string;
  isOpen: boolean;
  onClose: () => void;
}

export default function CommentsPanel({ taskId, isOpen, onClose }: CommentsPanelProps) {
  const [comments, setComments] = useState<Comment[]>([]);
  const [newComment, setNewComment] = useState('');

  useEffect(() => {
    if (isOpen) {
      const stored = localStorage.getItem(`bul_comments_${taskId}`);
      if (stored) {
        setComments(JSON.parse(stored).map((c: any) => ({
          ...c,
          timestamp: new Date(c.timestamp),
        })));
      }
    }
  }, [taskId, isOpen]);

  const addComment = () => {
    if (!newComment.trim()) return;

    const comment: Comment = {
      id: Date.now().toString(),
      text: newComment.trim(),
      author: 'Usuario', // En producción, usar usuario real
      timestamp: new Date(),
    };

    const updated = [...comments, comment];
    setComments(updated);
    localStorage.setItem(`bul_comments_${taskId}`, JSON.stringify(updated));
    setNewComment('');
  };

  const deleteComment = (id: string) => {
    const updated = comments.filter((c) => c.id !== id);
    setComments(updated);
    localStorage.setItem(`bul_comments_${taskId}`, JSON.stringify(updated));
  };

  if (!isOpen) return null;

  return (
    <motion.div
      initial={{ x: 300 }}
      animate={{ x: 0 }}
      exit={{ x: 300 }}
      className="fixed right-0 top-0 h-full w-96 bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 shadow-xl z-40 flex flex-col"
    >
      <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FiMessageSquare size={20} className="text-primary-600" />
          <h3 className="font-semibold text-gray-900 dark:text-white">Comentarios</h3>
        </div>
        <button onClick={onClose} className="btn-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M18 6L6 18" />
            <path d="M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {comments.length === 0 ? (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            <FiMessageSquare size={48} className="mx-auto mb-2 opacity-50" />
            <p>No hay comentarios aún</p>
          </div>
        ) : (
          comments.map((comment) => (
            <motion.div
              key={comment.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
            >
              <div className="flex items-start justify-between mb-2">
                <div>
                  <p className="text-sm font-medium text-gray-900 dark:text-white">
                    {comment.author}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {format(comment.timestamp, 'PPp')}
                  </p>
                </div>
                <button
                  onClick={() => deleteComment(comment.id)}
                  className="btn-icon text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20"
                >
                  <FiTrash2 size={14} />
                </button>
              </div>
              <p className="text-sm text-gray-700 dark:text-gray-300">{comment.text}</p>
            </motion.div>
          ))
        )}
      </div>

      <div className="p-4 border-t border-gray-200 dark:border-gray-700">
        <div className="flex gap-2">
          <input
            type="text"
            value={newComment}
            onChange={(e) => setNewComment(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                addComment();
              }
            }}
            placeholder="Escribe un comentario..."
            className="flex-1 input"
          />
          <button onClick={addComment} className="btn btn-primary">
            <FiSend size={18} />
          </button>
        </div>
      </div>
    </motion.div>
  );
}


