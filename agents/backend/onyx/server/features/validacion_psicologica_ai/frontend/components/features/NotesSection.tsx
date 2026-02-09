/**
 * Notes section component for validations
 */

'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, Textarea, Button } from '@/components/ui';
import { Save, Edit2, X } from 'lucide-react';
import { useLocalStorage } from '@/hooks/useLocalStorage';

export interface NotesSectionProps {
  validationId: string;
  className?: string;
}

export const NotesSection: React.FC<NotesSectionProps> = ({ validationId, className }) => {
  const [notes, setNotes] = useLocalStorage<string>(`notes-${validationId}`, '');
  const [isEditing, setIsEditing] = useState(false);
  const [tempNotes, setTempNotes] = useState(notes);

  const handleSave = () => {
    setNotes(tempNotes);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setTempNotes(notes);
    setIsEditing(false);
  };

  const handleEdit = () => {
    setTempNotes(notes);
    setIsEditing(true);
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Notas</CardTitle>
          {!isEditing ? (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleEdit}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  handleEdit();
                }
              }}
              aria-label="Editar notas"
              tabIndex={0}
            >
              <Edit2 className="h-4 w-4 mr-2" aria-hidden="true" />
              Editar
            </Button>
          ) : (
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleCancel}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    handleCancel();
                  }
                }}
                aria-label="Cancelar edición"
                tabIndex={0}
              >
                <X className="h-4 w-4 mr-2" aria-hidden="true" />
                Cancelar
              </Button>
              <Button
                variant="primary"
                size="sm"
                onClick={handleSave}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    handleSave();
                  }
                }}
                aria-label="Guardar notas"
                tabIndex={0}
              >
                <Save className="h-4 w-4 mr-2" aria-hidden="true" />
                Guardar
              </Button>
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {isEditing ? (
          <Textarea
            value={tempNotes}
            onChange={(e) => setTempNotes(e.target.value)}
            placeholder="Agrega tus notas sobre esta validación..."
            rows={6}
            aria-label="Notas de la validación"
          />
        ) : (
          <div className="min-h-[120px] p-3 border rounded-md bg-muted/50">
            {notes ? (
              <p className="text-sm whitespace-pre-wrap">{notes}</p>
            ) : (
              <p className="text-sm text-muted-foreground italic">
                No hay notas. Haz clic en "Editar" para agregar notas.
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};



