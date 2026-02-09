'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { FileText, Plus, Edit2, Trash2, Save, X } from 'lucide-react';
import toast from 'react-hot-toast';

interface NotesManagerProps {
  resourceId: string;
  resourceType: 'track' | 'analysis' | 'playlist';
}

export function NotesManager({ resourceId, resourceType }: NotesManagerProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [newNote, setNewNote] = useState('');
  const [editNote, setEditNote] = useState('');
  const queryClient = useQueryClient();

  const { data: notesData } = useQuery({
    queryKey: ['notes', resourceId, resourceType],
    queryFn: () => musicApiService.getNotes?.(resourceId, resourceType) || Promise.resolve({ notes: [] }),
  });

  const addNoteMutation = useMutation({
    mutationFn: (content: string) =>
      musicApiService.addNote?.(resourceId, resourceType, content) || Promise.resolve({}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notes', resourceId, resourceType] });
      toast.success('Nota agregada');
      setNewNote('');
      setIsEditing(false);
    },
    onError: () => {
      toast.error('Error al agregar nota');
    },
  });

  const updateNoteMutation = useMutation({
    mutationFn: ({ noteId, content }: { noteId: string; content: string }) =>
      musicApiService.updateNote?.(noteId, content) || Promise.resolve({}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notes', resourceId, resourceType] });
      toast.success('Nota actualizada');
      setEditingId(null);
      setEditNote('');
    },
    onError: () => {
      toast.error('Error al actualizar nota');
    },
  });

  const deleteNoteMutation = useMutation({
    mutationFn: (noteId: string) =>
      musicApiService.deleteNote?.(noteId) || Promise.resolve({}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notes', resourceId, resourceType] });
      toast.success('Nota eliminada');
    },
    onError: () => {
      toast.error('Error al eliminar nota');
    },
  });

  const notes = notesData?.notes || [];

  const handleAddNote = () => {
    if (!newNote.trim()) {
      toast.error('Ingresa una nota');
      return;
    }
    addNoteMutation.mutate(newNote.trim());
  };

  const handleStartEdit = (note: any) => {
    setEditingId(note.id);
    setEditNote(note.content);
  };

  const handleUpdateNote = (noteId: string) => {
    if (!editNote.trim()) {
      toast.error('La nota no puede estar vacía');
      return;
    }
    updateNoteMutation.mutate({ noteId, content: editNote.trim() });
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <FileText className="w-6 h-6 text-purple-300" />
          <h2 className="text-2xl font-semibold text-white">Notas</h2>
        </div>
        <button
          onClick={() => setIsEditing(!isEditing)}
          className="px-3 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          Nueva Nota
        </button>
      </div>

      {isEditing && (
        <div className="mb-4 p-4 bg-white/5 rounded-lg border border-white/10">
          <textarea
            value={newNote}
            onChange={(e) => setNewNote(e.target.value)}
            placeholder="Escribe tu nota aquí..."
            rows={4}
            className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400 mb-3"
          />
          <div className="flex gap-2">
            <button
              onClick={handleAddNote}
              disabled={addNoteMutation.isPending}
              className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
            >
              <Save className="w-4 h-4" />
              Guardar
            </button>
            <button
              onClick={() => {
                setIsEditing(false);
                setNewNote('');
              }}
              className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors flex items-center gap-2"
            >
              <X className="w-4 h-4" />
              Cancelar
            </button>
          </div>
        </div>
      )}

      <div className="space-y-3">
        {notes.length === 0 ? (
          <div className="text-center py-8">
            <FileText className="w-12 h-12 text-gray-500 mx-auto mb-2" />
            <p className="text-gray-400">No hay notas aún</p>
          </div>
        ) : (
          notes.map((note: any) => (
            <div
              key={note.id}
              className="p-4 bg-white/5 rounded-lg border border-white/10"
            >
              {editingId === note.id ? (
                <div>
                  <textarea
                    value={editNote}
                    onChange={(e) => setEditNote(e.target.value)}
                    rows={4}
                    className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-400 mb-3"
                  />
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleUpdateNote(note.id)}
                      disabled={updateNoteMutation.isPending}
                      className="px-3 py-1 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2 text-sm"
                    >
                      <Save className="w-3 h-3" />
                      Guardar
                    </button>
                    <button
                      onClick={() => {
                        setEditingId(null);
                        setEditNote('');
                      }}
                      className="px-3 py-1 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors flex items-center gap-2 text-sm"
                    >
                      <X className="w-3 h-3" />
                      Cancelar
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <p className="text-white whitespace-pre-wrap mb-2">{note.content}</p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-400">
                      {note.created_at
                        ? new Date(note.created_at).toLocaleDateString()
                        : 'Fecha desconocida'}
                    </span>
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleStartEdit(note)}
                        className="p-1 text-gray-400 hover:text-white transition-colors"
                      >
                        <Edit2 className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => deleteNoteMutation.mutate(note.id)}
                        className="p-1 text-gray-400 hover:text-red-300 transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
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


