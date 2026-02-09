/**
 * MessageBubble - Componente de Mensaje Individual
 * ==================================================
 * 
 * Componente reutilizable para mostrar mensajes individuales
 */

import React, { useState } from 'react';
import { Copy, Check, Edit, Trash2, ThumbsUp, ThumbsDown } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface MessageBubbleProps {
  message: {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: Date;
    error?: boolean;
  };
  isDarkMode: boolean;
  enableMarkdown?: boolean;
  enableActions?: boolean;
  onCopy?: (content: string) => void;
  onEdit?: (id: string, content: string) => void;
  onDelete?: (id: string) => void;
  onLike?: (id: string) => void;
  onDislike?: (id: string) => void;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  isDarkMode,
  enableMarkdown = true,
  enableActions = true,
  onCopy,
  onEdit,
  onDelete,
  onLike,
  onDislike,
}) => {
  const [copied, setCopied] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editedContent, setEditedContent] = useState(message.content);

  const handleCopy = () => {
    if (onCopy) {
      onCopy(message.content);
    } else {
      navigator.clipboard.writeText(message.content);
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleEdit = () => {
    if (isEditing && onEdit) {
      onEdit(message.id, editedContent);
      setIsEditing(false);
    } else {
      setIsEditing(true);
    }
  };

  const handleSave = () => {
    if (onEdit) {
      onEdit(message.id, editedContent);
      setIsEditing(false);
    }
  };

  const handleCancel = () => {
    setEditedContent(message.content);
    setIsEditing(false);
  };

  const messageClasses = message.error
    ? isDarkMode
      ? 'bg-red-900/20 border-red-500/50'
      : 'bg-red-50 border-red-200'
    : message.role === 'user'
    ? isDarkMode
      ? 'bg-blue-600 text-white'
      : 'bg-blue-500 text-white'
    : isDarkMode
    ? 'bg-gray-800 text-gray-100'
    : 'bg-gray-100 text-gray-900';

  return (
    <div className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-3xl rounded-lg p-4 border ${messageClasses} ${
          message.error ? 'border-current' : ''
        } shadow-sm hover:shadow-md transition-shadow`}
      >
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            {isEditing && message.role === 'user' ? (
              <div className="space-y-2">
                <textarea
                  value={editedContent}
                  onChange={(e) => setEditedContent(e.target.value)}
                  className={`w-full p-2 rounded ${
                    isDarkMode
                      ? 'bg-gray-700 text-gray-100 border-gray-600'
                      : 'bg-white text-gray-900 border-gray-300'
                  } border focus:outline-none focus:ring-2 ${
                    isDarkMode ? 'focus:ring-blue-500' : 'focus:ring-blue-400'
                  }`}
                  rows={4}
                  autoFocus
                />
                <div className="flex gap-2">
                  <button
                    onClick={handleSave}
                    className={`px-3 py-1 rounded text-sm ${
                      isDarkMode
                        ? 'bg-blue-600 hover:bg-blue-700 text-white'
                        : 'bg-blue-500 hover:bg-blue-600 text-white'
                    }`}
                  >
                    Guardar
                  </button>
                  <button
                    onClick={handleCancel}
                    className={`px-3 py-1 rounded text-sm ${
                      isDarkMode
                        ? 'bg-gray-700 hover:bg-gray-600 text-gray-100'
                        : 'bg-gray-200 hover:bg-gray-300 text-gray-900'
                    }`}
                  >
                    Cancelar
                  </button>
                </div>
              </div>
            ) : (
              <>
                {enableMarkdown && message.role === 'assistant' ? (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    className="prose prose-sm max-w-none"
                    components={{
                      code: ({ node, inline, ...props }: any) => (
                        <code
                          className={`${
                            inline
                              ? `px-1 py-0.5 rounded ${
                                  isDarkMode
                                    ? 'bg-gray-700 text-gray-200'
                                    : 'bg-gray-200 text-gray-800'
                                }`
                              : `block p-3 rounded-lg my-2 overflow-x-auto ${
                                  isDarkMode
                                    ? 'bg-gray-900 text-gray-200'
                                    : 'bg-gray-50 text-gray-800'
                                }`
                          }`}
                          {...props}
                        />
                      ),
                      p: ({ ...props }: any) => (
                        <p className="mb-2 last:mb-0" {...props} />
                      ),
                      a: ({ ...props }: any) => (
                        <a
                          className={`${
                            isDarkMode
                              ? 'text-blue-400 hover:text-blue-300'
                              : 'text-blue-600 hover:text-blue-700'
                          } underline`}
                          {...props}
                        />
                      ),
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                ) : (
                  <p className="whitespace-pre-wrap break-words">{message.content}</p>
                )}
                <p
                  className={`text-xs mt-2 ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}
                >
                  {message.timestamp.toLocaleTimeString()}
                </p>
              </>
            )}
          </div>

          {enableActions && !isEditing && (
            <div className="flex flex-col gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                onClick={handleCopy}
                className={`p-1.5 rounded transition-colors ${
                  isDarkMode
                    ? 'hover:bg-gray-700 text-gray-400 hover:text-gray-300'
                    : 'hover:bg-gray-200 text-gray-600 hover:text-gray-700'
                }`}
                title="Copiar"
              >
                {copied ? (
                  <Check className="w-4 h-4 text-green-500" />
                ) : (
                  <Copy className="w-4 h-4" />
                )}
              </button>

              {message.role === 'user' && onEdit && (
                <button
                  onClick={handleEdit}
                  className={`p-1.5 rounded transition-colors ${
                    isDarkMode
                      ? 'hover:bg-gray-700 text-gray-400 hover:text-gray-300'
                      : 'hover:bg-gray-200 text-gray-600 hover:text-gray-700'
                  }`}
                  title="Editar"
                >
                  <Edit className="w-4 h-4" />
                </button>
              )}

              {onDelete && (
                <button
                  onClick={() => onDelete(message.id)}
                  className={`p-1.5 rounded transition-colors ${
                    isDarkMode
                      ? 'hover:bg-red-900/20 text-red-400 hover:text-red-300'
                      : 'hover:bg-red-50 text-red-600 hover:text-red-700'
                  }`}
                  title="Eliminar"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              )}

              {message.role === 'assistant' && (
                <>
                  {onLike && (
                    <button
                      onClick={() => onLike(message.id)}
                      className={`p-1.5 rounded transition-colors ${
                        isDarkMode
                          ? 'hover:bg-gray-700 text-gray-400 hover:text-green-400'
                          : 'hover:bg-gray-200 text-gray-600 hover:text-green-600'
                      }`}
                      title="Me gusta"
                    >
                      <ThumbsUp className="w-4 h-4" />
                    </button>
                  )}
                  {onDislike && (
                    <button
                      onClick={() => onDislike(message.id)}
                      className={`p-1.5 rounded transition-colors ${
                        isDarkMode
                          ? 'hover:bg-gray-700 text-gray-400 hover:text-red-400'
                          : 'hover:bg-gray-200 text-gray-600 hover:text-red-600'
                      }`}
                      title="No me gusta"
                    >
                      <ThumbsDown className="w-4 h-4" />
                    </button>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;


