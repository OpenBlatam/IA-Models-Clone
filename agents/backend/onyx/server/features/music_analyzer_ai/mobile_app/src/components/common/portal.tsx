import React, { useEffect, useState } from 'react';
import { View, StyleSheet } from 'react-native';
import { Modal } from 'react-native';

interface PortalProps {
  children: React.ReactNode;
  visible: boolean;
}

/**
 * Portal component
 * Renders children in a modal overlay
 */
export function Portal({ children, visible }: PortalProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    return () => setMounted(false);
  }, []);

  if (!mounted || !visible) return null;

  return (
    <Modal visible={visible} transparent animationType="fade">
      <View style={styles.container}>{children}</View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

