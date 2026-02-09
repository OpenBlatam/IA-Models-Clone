import React from 'react';
import { View, TouchableOpacity, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../context/ThemeContext';

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  showFirstLast?: boolean;
  maxVisible?: number;
}

const Pagination: React.FC<PaginationProps> = ({
  currentPage,
  totalPages,
  onPageChange,
  showFirstLast = true,
  maxVisible = 5,
}) => {
  const { colors } = useTheme();

  const getVisiblePages = () => {
    const pages: (number | string)[] = [];
    const half = Math.floor(maxVisible / 2);

    let start = Math.max(1, currentPage - half);
    let end = Math.min(totalPages, start + maxVisible - 1);

    if (end - start < maxVisible - 1) {
      start = Math.max(1, end - maxVisible + 1);
    }

    if (showFirstLast && start > 1) {
      pages.push(1);
      if (start > 2) pages.push('...');
    }

    for (let i = start; i <= end; i++) {
      pages.push(i);
    }

    if (showFirstLast && end < totalPages) {
      if (end < totalPages - 1) pages.push('...');
      pages.push(totalPages);
    }

    return pages;
  };

  const visiblePages = getVisiblePages();

  return (
    <View style={styles.container}>
      <TouchableOpacity
        onPress={() => onPageChange(currentPage - 1)}
        disabled={currentPage === 1}
        style={[
          styles.button,
          {
            backgroundColor: colors.surface,
            opacity: currentPage === 1 ? 0.5 : 1,
          },
        ]}
        activeOpacity={0.7}
      >
        <Ionicons
          name="chevron-back"
          size={20}
          color={colors.text}
        />
      </TouchableOpacity>

      {visiblePages.map((page, index) => {
        if (page === '...') {
          return (
            <Text
              key={`ellipsis-${index}`}
              style={[styles.ellipsis, { color: colors.textSecondary }]}
            >
              ...
            </Text>
          );
        }

        const pageNum = page as number;
        const isActive = pageNum === currentPage;

        return (
          <TouchableOpacity
            key={pageNum}
            onPress={() => onPageChange(pageNum)}
            style={[
              styles.pageButton,
              {
                backgroundColor: isActive ? colors.primary : colors.surface,
              },
            ]}
            activeOpacity={0.7}
          >
            <Text
              style={[
                styles.pageText,
                {
                  color: isActive ? '#fff' : colors.text,
                  fontWeight: isActive ? '600' : '400',
                },
              ]}
            >
              {pageNum}
            </Text>
          </TouchableOpacity>
        );
      })}

      <TouchableOpacity
        onPress={() => onPageChange(currentPage + 1)}
        disabled={currentPage === totalPages}
        style={[
          styles.button,
          {
            backgroundColor: colors.surface,
            opacity: currentPage === totalPages ? 0.5 : 1,
          },
        ]}
        activeOpacity={0.7}
      >
        <Ionicons
          name="chevron-forward"
          size={20}
          color={colors.text}
        />
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  button: {
    width: 40,
    height: 40,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
  },
  pageButton: {
    minWidth: 40,
    height: 40,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 12,
  },
  pageText: {
    fontSize: 14,
  },
  ellipsis: {
    fontSize: 14,
    paddingHorizontal: 4,
  },
});

export default Pagination;

