import React, { memo, useMemo } from 'react';
import { OptimizedSectionList } from './optimized-section-list';
import { groupBy } from '../../utils/list-helpers';
import type { SectionListRenderItem, SectionListData } from 'react-native';

interface GroupedListProps<T> {
  data: T[];
  renderItem: SectionListRenderItem<T>;
  keyExtractor: (item: T, index: number) => string;
  groupBy: (item: T) => string;
  renderSectionHeader?: (section: { title: string; data: T[] }) => React.ReactNode;
  itemHeight?: number;
  sectionHeaderHeight?: number;
  estimatedItemHeight?: number;
}

function GroupedListComponent<T>({
  data,
  renderItem,
  keyExtractor,
  groupBy: groupByFn,
  renderSectionHeader,
  itemHeight,
  sectionHeaderHeight,
  estimatedItemHeight,
}: GroupedListProps<T>) {
  const sections = useMemo(() => {
    const grouped = groupBy(data, groupByFn);
    return Object.entries(grouped).map(([title, items]) => ({
      title,
      data: items,
    }));
  }, [data, groupByFn]);

  const sectionListData: SectionListData<T, { title: string; data: T[] }>[] = useMemo(
    () =>
      sections.map((section) => ({
        title: section.title,
        data: section.data,
      })),
    [sections]
  );

  const defaultRenderSectionHeader = useMemo(
    () => (info: { section: { title: string; data: T[] } }) => {
      if (renderSectionHeader) {
        return <>{renderSectionHeader(info.section)}</>;
      }
      return null;
    },
    [renderSectionHeader]
  );

  return (
    <OptimizedSectionList
      sections={sectionListData}
      renderItem={renderItem}
      keyExtractor={keyExtractor}
      renderSectionHeader={defaultRenderSectionHeader}
      itemHeight={itemHeight}
      sectionHeaderHeight={sectionHeaderHeight}
      estimatedItemHeight={estimatedItemHeight}
    />
  );
}

export const GroupedList = memo(GroupedListComponent) as <T>(
  props: GroupedListProps<T>
) => React.ReactElement;

