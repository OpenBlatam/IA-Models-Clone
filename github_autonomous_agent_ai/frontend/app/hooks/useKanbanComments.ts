import { useState } from 'react';

export interface Comment {
  id: string;
  text: string;
  timestamp: Date;
  author: string;
}

export function useKanbanComments() {
  const [taskComments, setTaskComments] = useState<Map<string, Comment[]>>(new Map());
  const [showComments, setShowComments] = useState<Map<string, boolean>>(new Map());

  const addComment = (taskId: string, text: string, author: string = 'Usuario') => {
    setTaskComments(prev => {
      const newComments = new Map(prev);
      const comments = newComments.get(taskId) || [];
      newComments.set(taskId, [
        ...comments,
        {
          id: `comment-${Date.now()}`,
          text,
          timestamp: new Date(),
          author,
        }
      ]);
      return newComments;
    });
  };

  const deleteComment = (taskId: string, commentId: string) => {
    setTaskComments(prev => {
      const newComments = new Map(prev);
      const comments = newComments.get(taskId) || [];
      newComments.set(taskId, comments.filter(c => c.id !== commentId));
      return newComments;
    });
  };

  const toggleComments = (taskId: string) => {
    setShowComments(prev => {
      const newMap = new Map(prev);
      newMap.set(taskId, !newMap.get(taskId));
      return newMap;
    });
  };

  return {
    taskComments,
    showComments,
    addComment,
    deleteComment,
    toggleComments,
  };
}

