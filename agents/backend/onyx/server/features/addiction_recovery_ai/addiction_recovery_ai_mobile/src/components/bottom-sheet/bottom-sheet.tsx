import React, { ReactNode, useCallback, useMemo, useRef } from 'react';
import BottomSheet, { BottomSheetView, BottomSheetBackdrop } from '@gorhom/bottom-sheet';
import { View, StyleSheet, ViewStyle } from 'react-native';
import { useColors } from '@/theme/colors';
import { SPACING } from '@/theme/spacing';

interface CustomBottomSheetProps {
  children: ReactNode;
  snapPoints?: (string | number)[];
  index?: number;
  onChange?: (index: number) => void;
  enablePanDownToClose?: boolean;
  style?: ViewStyle;
}

export function CustomBottomSheet({
  children,
  snapPoints = ['25%', '50%', '90%'],
  index = -1,
  onChange,
  enablePanDownToClose = true,
  style,
}: CustomBottomSheetProps): JSX.Element {
  const colors = useColors();
  const bottomSheetRef = useRef<BottomSheet>(null);

  const memoizedSnapPoints = useMemo(() => snapPoints, [snapPoints]);

  const handleSheetChanges = useCallback(
    (newIndex: number) => {
      onChange?.(newIndex);
    },
    [onChange]
  );

  const renderBackdrop = useCallback(
    (props: any) => (
      <BottomSheetBackdrop
        {...props}
        disappearsOnIndex={-1}
        appearsOnIndex={0}
        opacity={0.5}
      />
    ),
    []
  );

  return (
    <BottomSheet
      ref={bottomSheetRef}
      index={index}
      snapPoints={memoizedSnapPoints}
      onChange={handleSheetChanges}
      enablePanDownToClose={enablePanDownToClose}
      backgroundStyle={{ backgroundColor: colors.surface }}
      handleIndicatorStyle={{ backgroundColor: colors.border }}
      backdropComponent={renderBackdrop}
    >
      <BottomSheetView style={[styles.contentContainer, style]}>
        {children}
      </BottomSheetView>
    </BottomSheet>
  );
}

const styles = StyleSheet.create({
  contentContainer: {
    flex: 1,
    padding: SPACING.lg,
  },
});

