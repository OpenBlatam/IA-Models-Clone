import React, { useRef } from 'react';
import { View, StyleSheet, Animated, TouchableOpacity, Text } from 'react-native';
import { useTheme } from '../contexts/ThemeContext';
import { spacing, borderRadius, typography } from '../theme/colors';
import { hapticFeedback } from '../utils/haptics';

interface SwipeToDeleteProps {
  children: React.ReactNode;
  onDelete: () => void;
  deleteText?: string;
  threshold?: number;
}

export const SwipeToDelete: React.FC<SwipeToDeleteProps> = ({
  children,
  onDelete,
  deleteText = 'Eliminar',
  threshold = 100,
}) => {
  const { theme } = useTheme();
  const translateX = useRef(new Animated.Value(0)).current;

  const handleDelete = () => {
    hapticFeedback.success();
    onDelete();
    Animated.spring(translateX, {
      toValue: 0,
      useNativeDriver: true,
    }).start();
  };

  const panResponderInstance = React.useRef(
    React.PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,
      onPanResponderMove: (_, gestureState) => {
        if (gestureState.dx < 0) {
          translateX.setValue(gestureState.dx);
        }
      },
      onPanResponderRelease: (_, gestureState) => {
        if (gestureState.dx < -threshold) {
          handleDelete();
        } else {
          Animated.spring(translateX, {
            toValue: 0,
            useNativeDriver: true,
          }).start();
        }
      },
    })
  ).current;

  const deleteOpacity = translateX.interpolate({
    inputRange: [-threshold, 0],
    outputRange: [1, 0],
    extrapolate: 'clamp',
  });

  return (
    <View style={styles.container}>
      <View
        style={[
          styles.deleteContainer,
          {
            backgroundColor: theme.error,
          },
        ]}
      >
        <Animated.View style={{ opacity: deleteOpacity }}>
          <TouchableOpacity
            style={styles.deleteButton}
            onPress={handleDelete}
            activeOpacity={0.7}
          >
            <Text style={[styles.deleteText, { color: theme.surface }]}>
              {deleteText}
            </Text>
          </TouchableOpacity>
        </Animated.View>
      </View>
      <Animated.View
        style={[
          styles.content,
          {
            transform: [{ translateX }],
          },
        ]}
        {...panResponderInstance.panHandlers}
      >
        {children}
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'relative',
    overflow: 'hidden',
  },
  deleteContainer: {
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'flex-end',
    paddingRight: spacing.lg,
    zIndex: 0,
  },
  deleteButton: {
    padding: spacing.md,
  },
  deleteText: {
    ...typography.body,
    fontWeight: '600',
  },
  content: {
    backgroundColor: 'transparent',
    zIndex: 1,
  },
});

