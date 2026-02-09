import { useState, useCallback } from 'react';
import { Alert } from 'react-native';
import { useLogEntry } from '@/hooks/api';
import { useAuthStore } from '@/store/auth-store';
import type { Mood } from '@/types';

export function useProgressLogForm() {
  const { user } = useAuthStore();
  const [showLogForm, setShowLogForm] = useState(false);
  const [mood, setMood] = useState<Mood>('neutral');
  const [cravingsLevel, setCravingsLevel] = useState('5');
  const [notes, setNotes] = useState('');
  const logEntryMutation = useLogEntry();

  const toggleLogForm = useCallback(() => {
    setShowLogForm((prev) => !prev);
  }, []);

  const handleLogEntry = useCallback(async () => {
    if (!user?.user_id) {
      Alert.alert('Error', 'Usuario no identificado');
      return;
    }

    try {
      await logEntryMutation.mutateAsync({
        user_id: user.user_id,
        date: new Date().toISOString().split('T')[0],
        mood,
        cravings_level: parseInt(cravingsLevel, 10),
        notes: notes || undefined,
        triggers_encountered: [],
        consumed: false,
      });
      Alert.alert('Éxito', 'Entrada registrada correctamente');
      setShowLogForm(false);
      setNotes('');
      setCravingsLevel('5');
      setMood('neutral');
    } catch (error: any) {
      Alert.alert('Error', error.response?.data?.detail || 'Error al registrar entrada');
    }
  }, [user, mood, cravingsLevel, notes, logEntryMutation]);

  return {
    showLogForm,
    toggleLogForm,
    mood,
    setMood,
    cravingsLevel,
    setCravingsLevel,
    notes,
    setNotes,
    handleLogEntry,
    isLoading: logEntryMutation.isPending,
  };
}

