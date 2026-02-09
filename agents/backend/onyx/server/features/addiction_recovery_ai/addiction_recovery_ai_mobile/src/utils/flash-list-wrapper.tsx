import React, { memo } from 'react';
import { FlashList, FlashListProps } from '@shopify/flash-list';
import type { ListRenderItem } from '@shopify/flash-list';

interface FlashListWrapperProps<T> extends Omit<FlashListProps<T>, 'renderItem'> {
  data: T[];
  renderItem: ListRenderItem<T>;
  estimatedItemSize?: number;
}

function FlashListWrapperComponent<T>({
  data,
  renderItem,
  estimatedItemSize = 50,
  ...props
}: FlashListWrapperProps<T>): JSX.Element {
  return (
    <FlashList
      data={data}
      renderItem={renderItem}
      estimatedItemSize={estimatedItemSize}
      {...props}
    />
  );
}

export const FlashListWrapper = memo(FlashListWrapperComponent) as <T>(
  props: FlashListWrapperProps<T>
) => JSX.Element;

