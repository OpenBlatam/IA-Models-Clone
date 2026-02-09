/**
 * useConversations - Hook para Gestionar Conversaciones
 * ======================================================
 * 
 * Hook para gestionar múltiples conversaciones
 */

import { useState, useCallback, useEffect } from 'react';

export interface Conversation {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: Date;
  messageCount: number;
  isPinned?: boolean;
  isFolder?: boolean;
  folderId?: string;
  metadata?: Record<string, any>;
}

export interface UseConversationsOptions {
  storageKey?: string;
  autoSave?: boolean;
}

export interface UseConversationsReturn {
  conversations: Conversation[];
  activeConversationId: string | null;
  setActiveConversation: (id: string | null) => void;
  createConversation: (title?: string) => string;
  deleteConversation: (id: string) => void;
  renameConversation: (id: string, newTitle: string) => void;
  pinConversation: (id: string) => void;
  unpinConversation: (id: string) => void;
  updateConversation: (id: string, updates: Partial<Conversation>) => void;
  createFolder: (name: string) => string;
  moveToFolder: (conversationId: string, folderId: string) => void;
}

export const useConversations = (
  options: UseConversationsOptions = {}
): UseConversationsReturn => {
  const {
    storageKey = 'chat-conversations',
    autoSave = true,
  } = options;

  const [conversations, setConversations] = useState<Conversation[]>(() => {
    if (autoSave) {
      const saved = localStorage.getItem(storageKey);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          return parsed.map((conv: any) => ({
            ...conv,
            timestamp: new Date(conv.timestamp),
          }));
        } catch {
          return [];
        }
      }
    }
    return [];
  });

  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);

  // Guardar automáticamente
  useEffect(() => {
    if (autoSave && conversations.length > 0) {
      localStorage.setItem(storageKey, JSON.stringify(conversations));
    }
  }, [conversations, autoSave, storageKey]);

  const createConversation = useCallback((title?: string): string => {
    const id = `conv-${Date.now()}-${Math.random()}`;
    const newConversation: Conversation = {
      id,
      title: title || 'Nueva conversación',
      lastMessage: '',
      timestamp: new Date(),
      messageCount: 0,
      isPinned: false,
    };

    setConversations((prev) => [newConversation, ...prev]);
    setActiveConversationId(id);
    return id;
  }, []);

  const deleteConversation = useCallback((id: string) => {
    setConversations((prev) => prev.filter((conv) => conv.id !== id));
    if (activeConversationId === id) {
      setActiveConversationId(null);
    }
  }, [activeConversationId]);

  const renameConversation = useCallback((id: string, newTitle: string) => {
    setConversations((prev) =>
      prev.map((conv) =>
        conv.id === id ? { ...conv, title: newTitle } : conv
      )
    );
  }, []);

  const pinConversation = useCallback((id: string) => {
    setConversations((prev) =>
      prev.map((conv) =>
        conv.id === id ? { ...conv, isPinned: true } : conv
      )
    );
  }, []);

  const unpinConversation = useCallback((id: string) => {
    setConversations((prev) =>
      prev.map((conv) =>
        conv.id === id ? { ...conv, isPinned: false } : conv
      )
    );
  }, []);

  const updateConversation = useCallback((id: string, updates: Partial<Conversation>) => {
    setConversations((prev) =>
      prev.map((conv) =>
        conv.id === id ? { ...conv, ...updates } : conv
      )
    );
  }, []);

  const createFolder = useCallback((name: string): string => {
    const id = `folder-${Date.now()}-${Math.random()}`;
    const folder: Conversation = {
      id,
      title: name,
      lastMessage: '',
      timestamp: new Date(),
      messageCount: 0,
      isFolder: true,
      isPinned: false,
    };

    setConversations((prev) => [folder, ...prev]);
    return id;
  }, []);

  const moveToFolder = useCallback((conversationId: string, folderId: string) => {
    setConversations((prev) =>
      prev.map((conv) =>
        conv.id === conversationId ? { ...conv, folderId } : conv
      )
    );
  }, []);

  const setActiveConversation = useCallback((id: string | null) => {
    setActiveConversationId(id);
  }, []);

  return {
    conversations,
    activeConversationId,
    setActiveConversation,
    createConversation,
    deleteConversation,
    renameConversation,
    pinConversation,
    unpinConversation,
    updateConversation,
    createFolder,
    moveToFolder,
  };
};


