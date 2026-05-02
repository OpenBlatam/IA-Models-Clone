'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { Loader2, Sparkles } from 'lucide-react';

interface AINotesModalProps {
  isOpen: boolean;
  onClose: () => void;
  eventTitle: string;
  eventDescription?: string;
}

export function AINotesModal({ isOpen, onClose, eventTitle, eventDescription }: AINotesModalProps) {
  const [notes, setNotes] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);

  const generateAINotes = async () => {
    setIsGenerating(true);
    try {
      // Aquí iría la llamada a la API de IA
      // Por ahora, simulamos una respuesta
      await new Promise(resolve => setTimeout(resolve, 2000));
      const aiNotes = `Apuntes generados por IA para el evento "${eventTitle}":\n\n` +
        `1. Puntos clave:\n` +
        `   - Preparar materiales necesarios\n` +
        `   - Revisar agenda y objetivos\n` +
        `   - Confirmar asistencia de participantes\n\n` +
        `2. Sugerencias:\n` +
        `   - Documentar decisiones importantes\n` +
        `   - Asignar tareas de seguimiento\n` +
        `   - Programar próxima reunión si es necesario\n\n` +
        `3. Recordatorios:\n` +
        `   - Llevar documentación relevante\n` +
        `   - Preparar presentación si es necesario\n` +
        `   - Verificar conexión técnica`;
      
      setNotes(aiNotes);
    } catch (error) {
      console.error('Error generando notas:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            Apuntes de IA
          </DialogTitle>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium">Evento: {eventTitle}</h3>
              <Button
                variant="outline"
                size="sm"
                onClick={generateAINotes}
                disabled={isGenerating}
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generando...
                  </>
                ) : (
                  <>
                    <Sparkles className="mr-2 h-4 w-4" />
                    Generar Apuntes
                  </>
                )}
              </Button>
            </div>
            <Textarea
              placeholder="Los apuntes generados por IA aparecerán aquí..."
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              className="min-h-[300px] font-mono text-sm"
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cerrar
          </Button>
          <Button onClick={() => {
            // Aquí iría la lógica para guardar los apuntes
            onClose();
          }}>
            Guardar Apuntes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
} 