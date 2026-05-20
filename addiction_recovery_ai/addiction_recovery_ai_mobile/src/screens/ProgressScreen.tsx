import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Alert,
} from 'react-native';
import { Input, Button, ProgressCard, LoadingSpinner } from '@/components';
import { useProgress, useStats, useLogEntry } from '@/hooks/useApi';
import { useAuthStore } from '@/store/auth-store';
import { Picker } from '@react-native-picker/picker';

export const ProgressScreen: React.FC = () => {
  const { user } = useAuthStore();
  const [mood, setMood] = useState<'excellent' | 'good' | 'neutral' | 'poor' | 'terrible'>('neutral');
  const [cravingsLevel, setCravingsLevel] = useState('5');
  const [notes, setNotes] = useState('');
  const [showLogForm, setShowLogForm] = useState(false);

  const { data: progress, isLoading: progressLoading } = useProgress(user?.user_id || null);
  const { data: stats, isLoading: statsLoading } = useStats(user?.user_id || null);
  const logEntryMutation = useLogEntry();

  const handleLogEntry = async () => {
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
  };

  if (progressLoading || statsLoading) {
    return <LoadingSpinner />;
  }

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Mi Progreso</Text>
        <Button
          title={showLogForm ? 'Cancelar' : 'Registrar Entrada'}
          onPress={() => setShowLogForm(!showLogForm)}
          variant="outline"
          style={styles.logButton}
        />
      </View>

      {showLogForm && (
        <View style={styles.logForm}>
          <Text style={styles.formTitle}>Registrar Entrada del Día</Text>

          <View style={styles.pickerContainer}>
            <Text style={styles.label}>Estado de Ánimo</Text>
            <View style={styles.pickerWrapper}>
              <Picker
                selectedValue={mood}
                onValueChange={(value) => setMood(value)}
                style={styles.picker}
              >
                <Picker.Item label="Excelente" value="excellent" />
                <Picker.Item label="Bueno" value="good" />
                <Picker.Item label="Neutral" value="neutral" />
                <Picker.Item label="Malo" value="poor" />
                <Picker.Item label="Terrible" value="terrible" />
              </Picker>
            </View>
          </View>

          <Input
            label="Nivel de Ansias (0-10)"
            value={cravingsLevel}
            onChangeText={setCravingsLevel}
            keyboardType="number-pad"
            placeholder="5"
            style={styles.input}
          />

          <Input
            label="Notas (Opcional)"
            value={notes}
            onChangeText={setNotes}
            placeholder="¿Cómo te sientes hoy?"
            multiline
            numberOfLines={4}
            style={styles.input}
          />

          <Button
            title="Guardar Entrada"
            onPress={handleLogEntry}
            loading={logEntryMutation.isPending}
            style={styles.saveButton}
          />
        </View>
      )}

      {progress && (
        <View style={styles.cardsContainer}>
          <ProgressCard
            title="Días Sobrio"
            value={progress.days_sober}
            color="#34C759"
          />
          <ProgressCard
            title="Racha Actual"
            value={`${progress.streak_days} días`}
            color="#007AFF"
          />
          <ProgressCard
            title="Racha Más Larga"
            value={`${progress.longest_streak} días`}
            color="#5856D6"
          />
          <ProgressCard
            title="Progreso"
            value={`${progress.progress_percentage.toFixed(0)}%`}
            color="#FF9500"
          />
        </View>
      )}

      {stats && (
        <View style={styles.statsSection}>
          <Text style={styles.sectionTitle}>Estadísticas</Text>
          <View style={styles.statRow}>
            <Text style={styles.statLabel}>Días Totales:</Text>
            <Text style={styles.statValue}>{stats.total_days}</Text>
          </View>
          <View style={styles.statRow}>
            <Text style={styles.statLabel}>Recaídas:</Text>
            <Text style={styles.statValue}>{stats.relapse_count}</Text>
          </View>
          <View style={styles.statRow}>
            <Text style={styles.statLabel}>Ansias Promedio:</Text>
            <Text style={styles.statValue}>
              {stats.average_cravings.toFixed(1)}/10
            </Text>
          </View>
          {stats.most_common_triggers.length > 0 && (
            <View style={styles.triggersSection}>
              <Text style={styles.statLabel}>Triggers Más Comunes:</Text>
              {stats.most_common_triggers.map((trigger, index) => (
                <Text key={index} style={styles.triggerItem}>
                  • {trigger}
                </Text>
              ))}
            </View>
          )}
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#FFF',
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5EA',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  logButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  logForm: {
    backgroundColor: '#FFF',
    margin: 16,
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  formTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 16,
  },
  pickerContainer: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  pickerWrapper: {
    borderWidth: 1,
    borderColor: '#DDD',
    borderRadius: 8,
    overflow: 'hidden',
  },
  picker: {
    backgroundColor: '#FFF',
  },
  input: {
    marginBottom: 16,
  },
  saveButton: {
    marginTop: 8,
  },
  cardsContainer: {
    padding: 16,
  },
  statsSection: {
    backgroundColor: '#FFF',
    margin: 16,
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 16,
  },
  statRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E5EA',
  },
  statLabel: {
    fontSize: 16,
    color: '#666',
  },
  statValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  triggersSection: {
    marginTop: 16,
  },
  triggerItem: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
});

