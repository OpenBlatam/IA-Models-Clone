import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';

interface Step {
  title: string;
  description?: string;
  icon?: React.ReactNode;
}

interface ProgressStepsProps {
  steps: Step[];
  currentStep: number;
  onStepPress?: (stepIndex: number) => void;
  showConnector?: boolean;
}

export const ProgressSteps: React.FC<ProgressStepsProps> = ({
  steps,
  currentStep,
  onStepPress,
  showConnector = true,
}) => {
  const { theme } = useTheme();

  const getStepStatus = (index: number) => {
    if (index < currentStep) return 'completed';
    if (index === currentStep) return 'current';
    return 'pending';
  };

  return (
    <View style={styles.container}>
      {steps.map((step, index) => {
        const status = getStepStatus(index);
        const isCompleted = status === 'completed';
        const isCurrent = status === 'current';

        return (
          <View key={index} style={styles.stepContainer}>
            <View style={styles.stepContent}>
              {showConnector && index > 0 && (
                <View
                  style={[
                    styles.connector,
                    {
                      backgroundColor:
                        index <= currentStep ? theme.primary : theme.border,
                    },
                  ]}
                />
              )}
              <TouchableOpacity
                style={[
                  styles.stepCircle,
                  {
                    backgroundColor: isCompleted
                      ? theme.primary
                      : isCurrent
                      ? theme.primary + '20'
                      : theme.surfaceVariant,
                    borderColor: isCompleted || isCurrent ? theme.primary : theme.border,
                    borderWidth: isCurrent ? 2 : 1,
                  },
                ]}
                onPress={() => onStepPress?.(index)}
                disabled={!onStepPress}
                activeOpacity={0.7}
              >
                {isCompleted ? (
                  <Text style={[styles.checkmark, { color: theme.surface }]}>
                    ✓
                  </Text>
                ) : step.icon ? (
                  <View style={styles.iconContainer}>{step.icon}</View>
                ) : (
                  <Text
                    style={[
                      styles.stepNumber,
                      {
                        color:
                          isCurrent
                            ? theme.primary
                            : theme.textSecondary,
                      },
                    ]}
                  >
                    {index + 1}
                  </Text>
                )}
              </TouchableOpacity>
              <View style={styles.stepInfo}>
                <Text
                  style={[
                    styles.stepTitle,
                    {
                      color: isCurrent || isCompleted ? theme.text : theme.textSecondary,
                      fontWeight: isCurrent ? '600' : '400',
                    },
                  ]}
                >
                  {step.title}
                </Text>
                {step.description && (
                  <Text
                    style={[
                      styles.stepDescription,
                      { color: theme.textTertiary },
                    ]}
                  >
                    {step.description}
                  </Text>
                )}
              </View>
            </View>
          </View>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    paddingVertical: spacing.md,
  },
  stepContainer: {
    marginBottom: spacing.lg,
  },
  stepContent: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  connector: {
    position: 'absolute',
    left: 12,
    top: 32,
    width: 2,
    height: '100%',
    zIndex: 0,
  },
  stepCircle: {
    width: 24,
    height: 24,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.md,
    zIndex: 1,
  },
  checkmark: {
    fontSize: 14,
    fontWeight: '600',
  },
  iconContainer: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  stepNumber: {
    ...typography.bodySmall,
    fontWeight: '600',
  },
  stepInfo: {
    flex: 1,
    paddingTop: spacing.xs,
  },
  stepTitle: {
    ...typography.body,
    marginBottom: spacing.xs,
  },
  stepDescription: {
    ...typography.caption,
  },
});

