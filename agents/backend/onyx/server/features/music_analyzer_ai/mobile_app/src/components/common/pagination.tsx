import React from 'react';
import { View, TouchableOpacity, Text, StyleSheet } from 'react-native';
import { COLORS, SPACING, TYPOGRAPHY, BORDER_RADIUS } from '../../constants/config';

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  maxVisible?: number;
}

/**
 * Pagination component
 * Page navigation controls
 */
export function Pagination({
  currentPage,
  totalPages,
  onPageChange,
  maxVisible = 5,
}: PaginationProps) {
  const getVisiblePages = (): number[] => {
    const pages: number[] = [];
    const half = Math.floor(maxVisible / 2);

    let start = Math.max(1, currentPage - half);
    let end = Math.min(totalPages, start + maxVisible - 1);

    if (end - start < maxVisible - 1) {
      start = Math.max(1, end - maxVisible + 1);
    }

    for (let i = start; i <= end; i++) {
      pages.push(i);
    }

    return pages;
  };

  const visiblePages = getVisiblePages();

  return (
    <View style={styles.container}>
      <TouchableOpacity
        style={[styles.button, currentPage === 1 && styles.disabled]}
        onPress={() => onPageChange(currentPage - 1)}
        disabled={currentPage === 1}
        accessibilityLabel="Previous page"
        accessibilityRole="button"
      >
        <Text style={[styles.buttonText, currentPage === 1 && styles.disabledText]}>
          ‹
        </Text>
      </TouchableOpacity>

      {visiblePages.map((page) => (
        <TouchableOpacity
          key={page}
          style={[
            styles.pageButton,
            page === currentPage && styles.activePageButton,
          ]}
          onPress={() => onPageChange(page)}
          accessibilityLabel={`Page ${page}`}
          accessibilityRole="button"
          accessibilityState={{ selected: page === currentPage }}
        >
          <Text
            style={[
              styles.pageText,
              page === currentPage && styles.activePageText,
            ]}
          >
            {page}
          </Text>
        </TouchableOpacity>
      ))}

      <TouchableOpacity
        style={[styles.button, currentPage === totalPages && styles.disabled]}
        onPress={() => onPageChange(currentPage + 1)}
        disabled={currentPage === totalPages}
        accessibilityLabel="Next page"
        accessibilityRole="button"
      >
        <Text
          style={[
            styles.buttonText,
            currentPage === totalPages && styles.disabledText,
          ]}
        >
          ›
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: SPACING.xs,
  },
  button: {
    padding: SPACING.sm,
    minWidth: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  disabled: {
    opacity: 0.5,
  },
  buttonText: {
    ...TYPOGRAPHY.h3,
    color: COLORS.text,
  },
  disabledText: {
    color: COLORS.textSecondary,
  },
  pageButton: {
    minWidth: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: BORDER_RADIUS.md,
    backgroundColor: COLORS.surfaceLight,
  },
  activePageButton: {
    backgroundColor: COLORS.primary,
  },
  pageText: {
    ...TYPOGRAPHY.body,
    color: COLORS.text,
  },
  activePageText: {
    color: COLORS.surface,
    fontWeight: '600',
  },
});

