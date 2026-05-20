import React from 'react';
import { Editor } from '@tiptap/react';

interface EditorToolbarProps {
  editor: Editor | null;
}

const buttons = [
  { cmd: 'toggleBold', icon: <b>B</b>, label: 'Bold' },
  { cmd: 'toggleItalic', icon: <i>I</i>, label: 'Italic' },
  { cmd: 'toggleUnderline', icon: <u>U</u>, label: 'Underline' },
  { cmd: 'toggleBulletList', icon: <span style={{fontSize:'1.2em'}}>•≡</span>, label: 'List' },
  { cmd: 'setLink', icon: <span>🔗</span>, label: 'Link' },
  { cmd: 'setImage', icon: <span>🖼️</span>, label: 'Image' },
  { cmd: 'undo', icon: <span>↶</span>, label: 'Undo' },
  { cmd: 'redo', icon: <span>↷</span>, label: 'Redo' },
];

export default function EditorToolbar({ editor }: EditorToolbarProps) {
  if (!editor) return null;

  return (
    <div className="flex gap-3 p-3 bg-gray-50 rounded-t-xl border-b border-gray-200">
      <button
        className="rounded-full border border-indigo-900 text-indigo-900 w-10 h-10 flex items-center justify-center text-lg hover:bg-indigo-50 transition"
        onClick={() => editor.chain().focus().toggleBold().run()}
        aria-label="Bold"
        type="button"
      >
        <b>B</b>
      </button>
      <button
        className="rounded-full border border-indigo-900 text-indigo-900 w-10 h-10 flex items-center justify-center text-lg hover:bg-indigo-50 transition"
        onClick={() => editor.chain().focus().toggleItalic().run()}
        aria-label="Italic"
        type="button"
      >
        <i>I</i>
      </button>
      <button
        className="rounded-full border border-indigo-900 text-indigo-900 w-10 h-10 flex items-center justify-center text-lg hover:bg-indigo-50 transition"
        onClick={() => editor.chain().focus().toggleUnderline().run()}
        aria-label="Underline"
        type="button"
      >
        <u>U</u>
      </button>
      <button
        className="rounded-full border border-indigo-900 text-indigo-900 w-10 h-10 flex items-center justify-center text-lg hover:bg-indigo-50 transition"
        onClick={() => editor.chain().focus().toggleBulletList().run()}
        aria-label="List"
        type="button"
      >
        <span style={{fontSize:'1.2em'}}>•≡</span>
      </button>
      <button
        className="rounded-full border border-indigo-900 text-indigo-900 w-10 h-10 flex items-center justify-center text-lg hover:bg-indigo-50 transition"
        onClick={() => {
          const url = window.prompt('Enter URL');
          if (url) editor.chain().focus().setLink({ href: url }).run();
        }}
        aria-label="Link"
        type="button"
      >
        <span>🔗</span>
      </button>
      <button
        className="rounded-full border border-indigo-900 text-indigo-900 w-10 h-10 flex items-center justify-center text-lg hover:bg-indigo-50 transition"
        onClick={() => {
          const url = window.prompt('Enter image URL');
          if (url) editor.chain().focus().setImage({ src: url }).run();
        }}
        aria-label="Image"
        type="button"
      >
        <span>🖼️</span>
      </button>
      <button
        className="rounded-full border border-indigo-900 text-indigo-900 w-10 h-10 flex items-center justify-center text-lg hover:bg-indigo-50 transition"
        onClick={() => editor.chain().focus().undo().run()}
        aria-label="Undo"
        type="button"
      >
        <span>↶</span>
      </button>
      <button
        className="rounded-full border border-indigo-900 text-indigo-900 w-10 h-10 flex items-center justify-center text-lg hover:bg-indigo-50 transition"
        onClick={() => editor.chain().focus().redo().run()}
        aria-label="Redo"
        type="button"
      >
        <span>↷</span>
      </button>
    </div>
  );
} 