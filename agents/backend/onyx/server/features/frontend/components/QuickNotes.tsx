'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiFileText, FiPlus, FiX, FiEdit2, FiTrash2 } from 'react-icons/fi';
import { format } from 'date-fns';

interface Note {
  id: string;
  title: string;
  content: string;
  createdAt: Date;
  updatedAt: Date;
}

export default function QuickNotes() {
  const [notes, setNotes] = useState<Note[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [editingNote, setEditingNote] = useState<Note | null>(null);
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');

  useEffect(() => {
    const stored = localStorage.getItem('bul_quick_notes');
    if (stored) {
      setNotes(
        JSON.parse(stored).map((n: any) => ({
          ...n,
          createdAt: new Date(n.createdAt),
          updatedAt: new Date(n.updatedAt),
        }))
      );
    }
  }, []);

  const saveNotes = (updatedNotes: Note[]) => {
    setNotes(updatedNotes);
    localStorage.setItem('bul_quick_notes', JSON.stringify(updatedNotes));
  };

  const createNote = () => {
    const newNote: Note = {
      id: Date.now().toString(),
      title: title || 'Nota sin título',
      content,
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    saveNotes([...notes, newNote]);
    setTitle('');
    setContent('');
    setEditingNote(null);
  };

  const updateNote = (id: string) => {
    const updated = notes.map((n) =>
      n.id === id
        ? { ...n, title: title || 'Nota sin título', content, updatedAt: new Date() }
        : n
    );
    saveNotes(updated);
    setTitle('');
    setContent('');
    setEditingNote(null);
  };

  const deleteNote = (id: string) => {
    saveNotes(notes.filter((n) => n.id !== id));
  };

  const startEdit = (note: Note) => {
    setEditingNote(note);
    setTitle(note.title);
    setContent(note.content);
  };

  const cancelEdit = () => {
    setEditingNote(null);
    setTitle('');
    setContent('');
  };

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-20 right-4 bg-primary-600 text-white p-4 rounded-full shadow-lg hover:bg-primary-700 transition-colors z-40"
        title="Notas Rápidas"
      >
        <FiFileText size={24} />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
            onClick={() => setIsOpen(false)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-2xl w-full max-h-[80vh] flex flex-col"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Notas Rápidas
                </h3>
                <button onClick={() => setIsOpen(false)} className="btn-icon">
                  <FiX size={20} />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-4">
                {editingNote ? (
                  <div className="space-y-4">
                    <input
                      type="text"
                      value={title}
                      onChange={(e) => setTitle(e.target.value)}
                      placeholder="Título..."
                      className="input w-full"
                    />
                    <textarea
                      value={content}
                      onChange={(e) => setContent(e.target.value)}
                      placeholder="Contenido..."
                      className="textarea w-full min-h-[200px]"
                    />
                    <div className="flex gap-2">
                      <button
                        onClick={() => updateNote(editingNote.id)}
                        className="btn btn-primary flex-1"
                      >
                        Guardar
                      </button>
                      <button onClick={cancelEdit} className="btn btn-secondary flex-1">
                        Cancelar
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="mb-4">
                      <input
                        type="text"
                        value={title}
                        onChange={(e) => setTitle(e.target.value)}
                        placeholder="Título de la nueva nota..."
                        className="input w-full mb-2"
                      />
                      <textarea
                        value={content}
                        onChange={(e) => setContent(e.target.value)}
                        placeholder="Escribe tu nota aquí..."
                        className="textarea w-full min-h-[150px]"
                      />
                      <button
                        onClick={createNote}
                        disabled={!content.trim()}
                        className="btn btn-primary w-full mt-2"
                      >
                        <FiPlus size={18} className="mr-2" />
                        Crear Nota
                      </button>
                    </div>

                    <div className="space-y-2">
                      {notes.map((note) => (
                        <motion.div
                          key={note.id}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
                        >
                          <div className="flex items-start justify-between mb-2">
                            <div className="flex-1">
                              <h4 className="font-medium text-gray-900 dark:text-white">
                                {note.title}
                              </h4>
                              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                                {note.content.substring(0, 100)}
                                {note.content.length > 100 && '...'}
                              </p>
                              <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
                                {format(note.updatedAt, 'PPp')}
                              </p>
                            </div>
                            <div className="flex items-center gap-1 ml-2">
                              <button
                                onClick={() => startEdit(note)}
                                className="btn-icon text-primary-600"
                              >
                                <FiEdit2 size={16} />
                              </button>
                              <button
                                onClick={() => deleteNote(note.id)}
                                className="btn-icon text-red-600"
                              >
                                <FiTrash2 size={16} />
                              </button>
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}


