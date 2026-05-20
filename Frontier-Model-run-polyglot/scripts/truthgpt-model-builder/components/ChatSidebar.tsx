/**
 * ChatSidebar - Barra Lateral de Conversaciones
 * ===============================================
 * 
 * Componente para gestionar múltiples conversaciones
 */

import React, { useState } from 'react';
import { 
  MessageSquare, 
  Plus, 
  Search, 
  Trash2, 
  Edit, 
  Check, 
  X,
  Folder,
  Clock,
  Star
} from 'lucide-react';

interface Conversation {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: Date;
  isPinned?: boolean;
  isFolder?: boolean;
  folderId?: string;
}

interface ChatSidebarProps {
  conversations: Conversation[];
  activeConversationId?: string;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (id: string) => void;
  onRenameConversation: (id: string, newTitle: string) => void;
  onPinConversation?: (id: string) => void;
  onCreateFolder?: (name: string) => void;
  isDarkMode?: boolean;
  collapsed?: boolean;
  onToggleCollapse?: () => void;
}

export const ChatSidebar: React.FC<ChatSidebarProps> = ({
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  onRenameConversation,
  onPinConversation,
  onCreateFolder,
  isDarkMode = false,
  collapsed = false,
  onToggleCollapse,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');
  const [showNewFolder, setShowNewFolder] = useState(false);
  const [newFolderName, setNewFolderName] = useState('');

  const filteredConversations = conversations.filter(conv =>
    conv.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    conv.lastMessage.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleStartEdit = (conversation: Conversation) => {
    setEditingId(conversation.id);
    setEditValue(conversation.title);
  };

  const handleSaveEdit = () => {
    if (editingId && editValue.trim()) {
      onRenameConversation(editingId, editValue.trim());
      setEditingId(null);
      setEditValue('');
    }
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditValue('');
  };

  const handleCreateFolder = () => {
    if (newFolderName.trim() && onCreateFolder) {
      onCreateFolder(newFolderName.trim());
      setNewFolderName('');
      setShowNewFolder(false);
    }
  };

  const sortedConversations = [...filteredConversations].sort((a, b) => {
    if (a.isPinned && !b.isPinned) return -1;
    if (!a.isPinned && b.isPinned) return 1;
    return b.timestamp.getTime() - a.timestamp.getTime();
  });

  if (collapsed) {
    return (
      <div className={`${isDarkMode ? 'bg-gray-900' : 'bg-gray-100'} p-2`}>
        <button
          onClick={onToggleCollapse}
          className={`w-full p-2 rounded ${
            isDarkMode
              ? 'hover:bg-gray-800 text-gray-300'
              : 'hover:bg-gray-200 text-gray-600'
          }`}
        >
          <MessageSquare className="w-5 h-5 mx-auto" />
        </button>
      </div>
    );
  }

  return (
    <div
      className={`flex flex-col h-full ${
        isDarkMode ? 'bg-gray-900 text-gray-100' : 'bg-gray-50 text-gray-900'
      } border-r ${isDarkMode ? 'border-gray-800' : 'border-gray-200'}`}
      style={{ width: '300px' }}
    >
      {/* Header */}
      <div className={`p-4 border-b ${isDarkMode ? 'border-gray-800' : 'border-gray-200'}`}>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <MessageSquare className="w-5 h-5" />
            Conversaciones
          </h2>
          {onToggleCollapse && (
            <button
              onClick={onToggleCollapse}
              className={`p-1 rounded ${
                isDarkMode
                  ? 'hover:bg-gray-800 text-gray-400'
                  : 'hover:bg-gray-200 text-gray-600'
              }`}
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Nueva conversación */}
        <button
          onClick={onNewConversation}
          className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
            isDarkMode
              ? 'bg-blue-600 hover:bg-blue-700 text-white'
              : 'bg-blue-500 hover:bg-blue-600 text-white'
          }`}
        >
          <Plus className="w-4 h-4" />
          Nueva conversación
        </button>
      </div>

      {/* Búsqueda */}
      <div className="p-3 border-b border-gray-200 dark:border-gray-800">
        <div className="relative">
          <Search
            className={`absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 ${
              isDarkMode ? 'text-gray-400' : 'text-gray-500'
            }`}
          />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Buscar conversaciones..."
            className={`w-full pl-9 pr-3 py-2 rounded-lg text-sm ${
              isDarkMode
                ? 'bg-gray-800 text-gray-100 placeholder-gray-400 border-gray-700'
                : 'bg-white text-gray-900 placeholder-gray-500 border-gray-300'
            } border focus:outline-none focus:ring-2 ${
              isDarkMode ? 'focus:ring-blue-500' : 'focus:ring-blue-400'
            }`}
          />
        </div>
      </div>

      {/* Lista de conversaciones */}
      <div className="flex-1 overflow-y-auto">
        {sortedConversations.length === 0 ? (
          <div className={`p-8 text-center ${
            isDarkMode ? 'text-gray-400' : 'text-gray-500'
          }`}>
            <MessageSquare className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p className="text-sm">No hay conversaciones</p>
          </div>
        ) : (
          <div className="p-2">
            {sortedConversations.map((conversation) => (
              <div
                key={conversation.id}
                className={`group relative mb-1 rounded-lg p-3 cursor-pointer transition-colors ${
                  activeConversationId === conversation.id
                    ? isDarkMode
                      ? 'bg-blue-600/20 border-blue-500'
                      : 'bg-blue-50 border-blue-300'
                    : isDarkMode
                    ? 'hover:bg-gray-800'
                    : 'hover:bg-gray-100'
                } border ${
                  activeConversationId === conversation.id
                    ? isDarkMode
                      ? 'border-blue-500'
                      : 'border-blue-300'
                    : isDarkMode
                    ? 'border-transparent'
                    : 'border-transparent'
                }`}
                onClick={() => onSelectConversation(conversation.id)}
              >
                <div className="flex items-start gap-2">
                  {conversation.isPinned && (
                    <Star
                      className={`w-4 h-4 mt-1 flex-shrink-0 ${
                        isDarkMode ? 'text-yellow-400' : 'text-yellow-600'
                      }`}
                      fill="currentColor"
                    />
                  )}
                  <div className="flex-1 min-w-0">
                    {editingId === conversation.id ? (
                      <div className="flex items-center gap-1">
                        <input
                          type="text"
                          value={editValue}
                          onChange={(e) => setEditValue(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleSaveEdit();
                            if (e.key === 'Escape') handleCancelEdit();
                          }}
                          className={`flex-1 px-2 py-1 rounded text-sm ${
                            isDarkMode
                              ? 'bg-gray-800 text-gray-100 border-gray-700'
                              : 'bg-white text-gray-900 border-gray-300'
                          } border focus:outline-none focus:ring-2 ${
                            isDarkMode ? 'focus:ring-blue-500' : 'focus:ring-blue-400'
                          }`}
                          autoFocus
                          onClick={(e) => e.stopPropagation()}
                        />
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleSaveEdit();
                          }}
                          className={`p-1 rounded ${
                            isDarkMode
                              ? 'hover:bg-gray-700 text-green-400'
                              : 'hover:bg-gray-200 text-green-600'
                          }`}
                        >
                          <Check className="w-4 h-4" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCancelEdit();
                          }}
                          className={`p-1 rounded ${
                            isDarkMode
                              ? 'hover:bg-gray-700 text-red-400'
                              : 'hover:bg-gray-200 text-red-600'
                          }`}
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    ) : (
                      <>
                        <h3 className="font-medium text-sm truncate">
                          {conversation.title}
                        </h3>
                        <p
                          className={`text-xs mt-1 truncate ${
                            isDarkMode ? 'text-gray-400' : 'text-gray-600'
                          }`}
                        >
                          {conversation.lastMessage}
                        </p>
                        <div className="flex items-center gap-2 mt-1">
                          <Clock className={`w-3 h-3 ${
                            isDarkMode ? 'text-gray-500' : 'text-gray-400'
                          }`} />
                          <span
                            className={`text-xs ${
                              isDarkMode ? 'text-gray-500' : 'text-gray-400'
                            }`}
                          >
                            {conversation.timestamp.toLocaleDateString()}
                          </span>
                        </div>
                      </>
                    )}
                  </div>
                </div>

                {/* Acciones */}
                {editingId !== conversation.id && (
                  <div className="absolute right-2 top-2 opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
                    {onPinConversation && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onPinConversation(conversation.id);
                        }}
                        className={`p-1 rounded ${
                          isDarkMode
                            ? 'hover:bg-gray-700 text-gray-400'
                            : 'hover:bg-gray-200 text-gray-600'
                        }`}
                        title="Fijar"
                      >
                        <Star
                          className={`w-4 h-4 ${
                            conversation.isPinned
                              ? isDarkMode
                                ? 'text-yellow-400 fill-yellow-400'
                                : 'text-yellow-600 fill-yellow-600'
                              : ''
                          }`}
                        />
                      </button>
                    )}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleStartEdit(conversation);
                      }}
                      className={`p-1 rounded ${
                        isDarkMode
                          ? 'hover:bg-gray-700 text-gray-400'
                          : 'hover:bg-gray-200 text-gray-600'
                      }`}
                      title="Renombrar"
                    >
                      <Edit className="w-4 h-4" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteConversation(conversation.id);
                      }}
                      className={`p-1 rounded ${
                        isDarkMode
                          ? 'hover:bg-red-900/20 text-red-400'
                          : 'hover:bg-red-50 text-red-600'
                      }`}
                      title="Eliminar"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Crear carpeta */}
      {showNewFolder && (
        <div className={`p-3 border-t ${isDarkMode ? 'border-gray-800' : 'border-gray-200'}`}>
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={newFolderName}
              onChange={(e) => setNewFolderName(e.target.value)}
              placeholder="Nombre de carpeta..."
              className={`flex-1 px-2 py-1 rounded text-sm ${
                isDarkMode
                  ? 'bg-gray-800 text-gray-100 border-gray-700'
                  : 'bg-white text-gray-900 border-gray-300'
              } border focus:outline-none`}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleCreateFolder();
                if (e.key === 'Escape') {
                  setShowNewFolder(false);
                  setNewFolderName('');
                }
              }}
              autoFocus
            />
            <button
              onClick={handleCreateFolder}
              className={`p-1 rounded ${
                isDarkMode ? 'text-green-400' : 'text-green-600'
              }`}
            >
              <Check className="w-4 h-4" />
            </button>
            <button
              onClick={() => {
                setShowNewFolder(false);
                setNewFolderName('');
              }}
              className={`p-1 rounded ${
                isDarkMode ? 'text-red-400' : 'text-red-600'
              }`}
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Botón crear carpeta */}
      {onCreateFolder && !showNewFolder && (
        <div className={`p-3 border-t ${isDarkMode ? 'border-gray-800' : 'border-gray-200'}`}>
          <button
            onClick={() => setShowNewFolder(true)}
            className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
              isDarkMode
                ? 'hover:bg-gray-800 text-gray-300'
                : 'hover:bg-gray-200 text-gray-600'
            }`}
          >
            <Folder className="w-4 h-4" />
            Nueva carpeta
          </button>
        </div>
      )}
    </div>
  );
};

export default ChatSidebar;


