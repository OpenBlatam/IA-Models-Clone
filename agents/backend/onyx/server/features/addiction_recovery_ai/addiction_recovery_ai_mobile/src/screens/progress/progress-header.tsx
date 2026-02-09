import React from 'react';
import { View, Text } from 'react-native';
import { Button } from '@/components';
import { useColors } from '@/theme/colors';
import { useProgressStyles } from './progress-screen.styles';
import { useProgressLogForm } from './use-progress-log-form';

export function ProgressHeader(): JSX.Element {
  const colors = useColors();
  const styles = useProgressStyles(colors);
  const { showLogForm, toggleLogForm } = useProgressLogForm();

  return (
    <View style={styles.header}>
      <Text style={styles.title}>Mi Progreso</Text>
      <Button
        title={showLogForm ? 'Cancelar' : 'Registrar Entrada'}
        onPress={toggleLogForm}
        variant="outline"
        style={styles.logButton}
      />
    </View>
  );
}

