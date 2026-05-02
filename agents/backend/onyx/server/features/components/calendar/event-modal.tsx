'use client';

import { useState } from 'react';
import { format } from 'date-fns';
import { es } from 'date-fns/locale';
import { CalendarBlank, MapPin, Clock as ClockPhosphor, Tag, PencilSimple } from 'phosphor-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Calendar } from '@/components/ui/calendar';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { cn } from '@/lib/utils';
import { TimePickerClient } from './TimePickerClient';
import { motion } from 'framer-motion';

interface EventModalProps {
  isOpen: boolean;
  onClose: () => void;
  event?: {
    id?: string;
    title: string;
    start: Date;
    end: Date;
    description?: string;
    location?: string;
    color?: string;
  };
  onSave: (event: any) => void;
}

const EVENT_COLORS = [
  { id: '1', name: 'Lavanda', value: '#7986cb' },
  { id: '2', name: 'Sage', value: '#33b679' },
  { id: '3', name: 'Uva', value: '#8e24aa' },
  { id: '4', name: 'Flamingo', value: '#e67c73' },
  { id: '5', name: 'Plátano', value: '#f6c026' },
  { id: '6', name: 'Mandarina', value: '#f5511d' },
  { id: '7', name: 'Pavo Real', value: '#039be5' },
  { id: '8', name: 'Grafito', value: '#616161' },
  { id: '9', name: 'Arándano', value: '#3f51b5' },
  { id: '10', name: 'Albahaca', value: '#0b8043' },
  { id: '11', name: 'Tomate', value: '#d60000' },
];

export function EventModal({ isOpen, onClose, event, onSave }: EventModalProps) {
  const [title, setTitle] = useState(event?.title || '');
  const [description, setDescription] = useState(event?.description || '');
  const [location, setLocation] = useState(event?.location || '');
  const [startDate, setStartDate] = useState<Date>(event?.start || new Date());
  const [endDate, setEndDate] = useState<Date>(event?.end || new Date());
  const [color, setColor] = useState(event?.color || EVENT_COLORS[0].value);

  // Helpers para hora
  const getTimeString = (date: Date) => date.toISOString().slice(11, 16);
  const handleStartTimeChange = (value: string) => {
    const [hours, minutes] = value.split(':').map(Number);
    const newDate = new Date(startDate);
    newDate.setHours(hours, minutes, 0, 0);
    setStartDate(newDate);
  };
  const handleEndTimeChange = (value: string) => {
    const [hours, minutes] = value.split(':').map(Number);
    const newDate = new Date(endDate);
    newDate.setHours(hours, minutes, 0, 0);
    setEndDate(newDate);
  };

  const handleSave = () => {
    if (!title.trim()) return; // No guardar si no hay título
    onSave({
      id: event?.id || String(new Date().getTime()),
      title,
      description,
      location,
      start: startDate,
      end: endDate,
      color,
    });
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[520px] bg-white/80 dark:bg-muted/90 rounded-2xl shadow-2xl border-0 animate-in fade-in backdrop-blur-xl p-0 overflow-hidden">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="p-8"
        >
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-2xl font-bold text-primary mb-4">
              <PencilSimple size={28} className="text-violet-500" />
              {event ? 'Editar Evento' : 'Nuevo Evento'}
            </DialogTitle>
          </DialogHeader>
          <div className="grid gap-6">
            <div className="grid gap-2">
              <Input
                placeholder="Título del evento"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className="text-lg font-semibold px-4 py-3 rounded-xl border border-border focus:ring-2 focus:ring-primary/30"
              />
            </div>
            <div className="flex flex-col md:flex-row gap-4">
              <div className="flex-1 space-y-2">
                <label className="text-sm font-medium flex items-center gap-1"><ClockPhosphor size={18} />Inicio</label>
                <div className="flex gap-2 items-center">
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button
                        variant="outline"
                        className={cn(
                          "w-full justify-start text-left font-normal px-3 py-2 rounded-xl border border-border bg-white dark:bg-muted",
                          !startDate && "text-muted-foreground"
                        )}
                      >
                        <CalendarBlank size={18} className="mr-2 text-violet-500" />
                        {startDate ? format(startDate, "PPP", { locale: es }) : "Seleccionar fecha"}
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-0">
                      <Calendar
                        mode="single"
                        selected={startDate}
                        onSelect={(date) => date && setStartDate(date)}
                        initialFocus
                      />
                    </PopoverContent>
                  </Popover>
                  <TimePickerClient
                    onChange={handleStartTimeChange}
                    value={getTimeString(startDate)}
                    theme="light"
                    className="w-[110px]"
                  />
                </div>
              </div>
              <div className="flex-1 space-y-2">
                <label className="text-sm font-medium flex items-center gap-1"><ClockPhosphor size={18} />Fin</label>
                <div className="flex gap-2 items-center">
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button
                        variant="outline"
                        className={cn(
                          "w-full justify-start text-left font-normal px-3 py-2 rounded-xl border border-border bg-white dark:bg-muted",
                          !endDate && "text-muted-foreground"
                        )}
                      >
                        <CalendarBlank size={18} className="mr-2 text-violet-500" />
                        {endDate ? format(endDate, "PPP", { locale: es }) : "Seleccionar fecha"}
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-0">
                      <Calendar
                        mode="single"
                        selected={endDate}
                        onSelect={(date) => date && setEndDate(date)}
                        initialFocus
                      />
                    </PopoverContent>
                  </Popover>
                  <TimePickerClient
                    onChange={handleEndTimeChange}
                    value={getTimeString(endDate)}
                    theme="light"
                    className="w-[110px]"
                  />
                </div>
              </div>
            </div>
            <hr className="my-2 border-t border-border/40" />
            <div className="space-y-2">
              <label className="text-sm font-medium flex items-center gap-1"><Tag size={16} />Descripción</label>
              <Textarea
                placeholder="Añade una descripción..."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="min-h-[90px] px-4 py-2 rounded-xl border border-border focus:ring-2 focus:ring-primary/30"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium flex items-center gap-1"><MapPin size={16} />Ubicación</label>
              <div className="flex items-center gap-2">
                <Input
                  placeholder="Añade una ubicación..."
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  className="px-4 py-2 rounded-xl border border-border"
                />
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium flex items-center gap-1"><Tag size={16} />Color</label>
              <div className="flex flex-wrap gap-2">
                {EVENT_COLORS.map((c) => (
                  <button
                    key={c.id}
                    className={cn(
                      "h-7 w-7 rounded-full border-2 border-white shadow transition-all hover:scale-110",
                      color === c.value && "ring-2 ring-offset-2 ring-primary"
                    )}
                    style={{ backgroundColor: c.value }}
                    onClick={() => setColor(c.value)}
                    type="button"
                  />
                ))}
              </div>
            </div>
          </div>
          <DialogFooter className="flex flex-row gap-4 justify-end mt-8">
            <Button 
              variant="outline" 
              onClick={onClose} 
              className="rounded-xl px-6 py-2 w-full max-w-[160px] font-semibold border border-border"
            >
              Cancelar
            </Button>
            <motion.button
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
              onClick={handleSave}
              className="rounded-xl px-6 py-2 w-full max-w-[160px] font-semibold shadow bg-gradient-to-r from-violet-500 to-blue-500 text-white hover:from-violet-600 hover:to-blue-600 transition-all focus:outline-none focus:ring-2 focus:ring-violet-400"
            >
              {event ? 'Guardar cambios' : 'Crear evento'}
            </motion.button>
          </DialogFooter>
        </motion.div>
      </DialogContent>
    </Dialog>
  );
}    