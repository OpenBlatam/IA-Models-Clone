import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { TextField } from './TextField';

interface PasswordInputProps {
  label?: string;
  value: string;
  onChangeText: (text: string) => void;
  error?: string;
  placeholder?: string;
  showStrengthIndicator?: boolean;
}

export const PasswordInput: React.FC<PasswordInputProps> = ({
  label = 'Contraseña',
  value,
  onChangeText,
  error,
  placeholder = 'Ingresa tu contraseña',
  showStrengthIndicator = true,
}) => {
  const { theme } = useTheme();
  const [isVisible, setIsVisible] = useState(false);

  const getPasswordStrength = (password: string) => {
    if (password.length === 0) return { strength: 0, label: '', color: theme.border };
    if (password.length < 6) return { strength: 1, label: 'Débil', color: theme.error };
    
    let strength = 0;
    if (password.length >= 8) strength++;
    if (/[a-z]/.test(password) && /[A-Z]/.test(password)) strength++;
    if (/\d/.test(password)) strength++;
    if (/[^a-zA-Z\d]/.test(password)) strength++;

    if (strength <= 1) return { strength: 1, label: 'Débil', color: theme.error };
    if (strength === 2) return { strength: 2, label: 'Media', color: '#FF9800' };
    if (strength === 3) return { strength: 3, label: 'Fuerte', color: '#4CAF50' };
    return { strength: 4, label: 'Muy Fuerte', color: '#4CAF50' };
  };

  const strength = getPasswordStrength(value);

  return (
    <View>
      <TextField
        label={label}
        value={value}
        onChangeText={onChangeText}
        error={error}
        placeholder={placeholder}
        secureTextEntry={!isVisible}
        rightIcon={
          <TouchableOpacity
            onPress={() => setIsVisible(!isVisible)}
            hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
          >
            <Text style={{ color: theme.textSecondary, fontSize: 18 }}>
              {isVisible ? '👁️' : '👁️‍🗨️'}
            </Text>
          </TouchableOpacity>
        }
      />
      {showStrengthIndicator && value.length > 0 && (
        <View style={styles.strengthContainer}>
          <View style={styles.strengthBar}>
            <View
              style={[
                styles.strengthFill,
                {
                  width: `${(strength.strength / 4) * 100}%`,
                  backgroundColor: strength.color,
                },
              ]}
            />
          </View>
          <Text style={[styles.strengthLabel, { color: strength.color }]}>
            {strength.label}
          </Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  strengthContainer: {
    marginTop: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  strengthBar: {
    flex: 1,
    height: 4,
    backgroundColor: '#E0E0E0',
    borderRadius: 2,
    overflow: 'hidden',
  },
  strengthFill: {
    height: '100%',
    borderRadius: 2,
  },
  strengthLabel: {
    fontSize: 12,
    fontWeight: '600',
    minWidth: 60,
  },
});

