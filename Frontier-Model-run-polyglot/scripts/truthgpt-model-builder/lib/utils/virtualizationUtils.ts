/**
 * Utilidades de Virtualización
 * ============================
 * 
 * Funciones para virtualización de listas
 */

export interface VirtualItem {
  index: number
  start: number
  end: number
  size: number
}

export interface VirtualizationConfig {
  itemCount: number
  itemSize: number | ((index: number) => number)
  containerSize: number
  overscan?: number
  scrollOffset?: number
}

/**
 * Calcula items virtuales visibles
 */
export function calculateVirtualItems(
  config: VirtualizationConfig
): VirtualItem[] {
  const {
    itemCount,
    itemSize,
    containerSize,
    overscan = 5,
    scrollOffset = 0
  } = config

  const items: VirtualItem[] = []
  let currentOffset = 0
  let startIndex = 0
  let endIndex = itemCount - 1

  // Encontrar startIndex
  for (let i = 0; i < itemCount; i++) {
    const size = typeof itemSize === 'function' ? itemSize(i) : itemSize
    const nextOffset = currentOffset + size

    if (nextOffset > scrollOffset) {
      startIndex = Math.max(0, i - overscan)
      break
    }

    currentOffset = nextOffset
  }

  // Encontrar endIndex
  currentOffset = 0
  for (let i = 0; i < itemCount; i++) {
    const size = typeof itemSize === 'function' ? itemSize(i) : itemSize
    currentOffset += size

    if (currentOffset > scrollOffset + containerSize) {
      endIndex = Math.min(itemCount - 1, i + overscan)
      break
    }
  }

  // Calcular items visibles
  currentOffset = 0
  for (let i = 0; i < startIndex; i++) {
    const size = typeof itemSize === 'function' ? itemSize(i) : itemSize
    currentOffset += size
  }

  for (let i = startIndex; i <= endIndex; i++) {
    const size = typeof itemSize === 'function' ? itemSize(i) : itemSize
    items.push({
      index: i,
      start: currentOffset,
      end: currentOffset + size,
      size
    })
    currentOffset += size
  }

  return items
}

/**
 * Calcula el tamaño total de una lista virtualizada
 */
export function calculateTotalSize(
  itemCount: number,
  itemSize: number | ((index: number) => number)
): number {
  if (typeof itemSize === 'number') {
    return itemCount * itemSize
  }

  let total = 0
  for (let i = 0; i < itemCount; i++) {
    total += itemSize(i)
  }
  return total
}

/**
 * Encuentra el índice de un item basado en el offset
 */
export function findItemIndex(
  offset: number,
  itemCount: number,
  itemSize: number | ((index: number) => number)
): number {
  let currentOffset = 0

  for (let i = 0; i < itemCount; i++) {
    const size = typeof itemSize === 'function' ? itemSize(i) : itemSize
    const nextOffset = currentOffset + size

    if (offset >= currentOffset && offset < nextOffset) {
      return i
    }

    currentOffset = nextOffset
  }

  return itemCount - 1
}






