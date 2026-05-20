'use client';

import { useState, useRef } from 'react';
import { FiBold, FiItalic, FiList, FiLink, FiCode, FiType } from 'react-icons/fi';

interface RichTextEditorProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
}

export default function RichTextEditor({
  value,
  onChange,
  placeholder = 'Escribe tu contenido...',
  className = '',
}: RichTextEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const insertText = (before: string, after: string = '') => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const selectedText = value.substring(start, end);
    const newText =
      value.substring(0, start) +
      before +
      selectedText +
      after +
      value.substring(end);

    onChange(newText);

    setTimeout(() => {
      textarea.focus();
      const newCursorPos = start + before.length + selectedText.length;
      textarea.setSelectionRange(newCursorPos, newCursorPos);
    }, 0);
  };

  const toolbarActions = [
    {
      icon: FiBold,
      label: 'Negrita',
      action: () => insertText('**', '**'),
    },
    {
      icon: FiItalic,
      label: 'Cursiva',
      action: () => insertText('_', '_'),
    },
    {
      icon: FiCode,
      label: 'Código',
      action: () => insertText('`', '`'),
    },
    {
      icon: FiList,
      label: 'Lista',
      action: () => insertText('- '),
    },
    {
      icon: FiLink,
      label: 'Enlace',
      action: () => insertText('[', '](url)'),
    },
    {
      icon: FiType,
      label: 'Título',
      action: () => insertText('# '),
    },
  ];

  return (
    <div className={`border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden ${className}`}>
      <div className="flex items-center gap-1 p-2 bg-gray-50 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600">
        {toolbarActions.map((action, index) => (
          <button
            key={index}
            onClick={action.action}
            className="btn-icon p-2"
            title={action.label}
          >
            <action.icon size={16} />
          </button>
        ))}
      </div>
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full p-4 border-none outline-none resize-none bg-white dark:bg-gray-800 text-gray-900 dark:text-white min-h-[200px]"
        rows={6}
      />
    </div>
  );
}


