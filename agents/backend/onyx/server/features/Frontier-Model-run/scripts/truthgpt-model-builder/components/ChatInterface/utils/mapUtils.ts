/**
 * Utilidades compartidas para operaciones con Maps
 * Elimina código duplicado entre servicios
 */

/**
 * Agrega un elemento a un array en un Map
 */
export function addToMapArray<T>(
  map: Map<string, T[]>,
  key: string,
  item: T
): Map<string, T[]> {
  const newMap = new Map(map)
  const existing = newMap.get(key) || []
  newMap.set(key, [...existing, item])
  return newMap
}

/**
 * Elimina un elemento de un array en un Map por índice
 */
export function removeFromMapArray<T>(
  map: Map<string, T[]>,
  key: string,
  index: number
): Map<string, T[]> {
  const newMap = new Map(map)
  const existing = newMap.get(key) || []
  const filtered = existing.filter((_, i) => i !== index)
  
  if (filtered.length === 0) {
    newMap.delete(key)
  } else {
    newMap.set(key, filtered)
  }
  
  return newMap
}

/**
 * Obtiene un array de un Map o retorna array vacío
 */
export function getFromMapArray<T>(
  map: Map<string, T[]>,
  key: string
): T[] {
  return map.get(key) || []
}

/**
 * Verifica si un Map tiene un key con array no vacío
 */
export function hasInMapArray<T>(
  map: Map<string, T[]>,
  key: string
): boolean {
  return map.has(key) && map.get(key)!.length > 0
}

/**
 * Cuenta elementos en un array de un Map
 */
export function countInMapArray<T>(
  map: Map<string, T[]>,
  key: string
): number {
  return getFromMapArray(map, key).length
}

/**
 * Agrega un elemento a un Map
 */
export function addToMap<T>(
  map: Map<string, T>,
  key: string,
  item: T
): Map<string, T> {
  const newMap = new Map(map)
  newMap.set(key, item)
  return newMap
}

/**
 * Elimina un elemento de un Map
 */
export function removeFromMap<T>(
  map: Map<string, T>,
  key: string
): Map<string, T> {
  const newMap = new Map(map)
  newMap.delete(key)
  return newMap
}

/**
 * Obtiene un elemento de un Map
 */
export function getFromMap<T>(
  map: Map<string, T>,
  key: string
): T | undefined {
  return map.get(key)
}

/**
 * Verifica si un Map tiene un key
 */
export function hasInMap<T>(
  map: Map<string, T>,
  key: string
): boolean {
  return map.has(key)
}

/**
 * Filtra un Map por predicado
 */
export function filterMap<T>(
  map: Map<string, T>,
  predicate: (value: T, key: string) => boolean
): Map<string, T> {
  const filtered = new Map<string, T>()
  map.forEach((value, key) => {
    if (predicate(value, key)) {
      filtered.set(key, value)
    }
  })
  return filtered
}

/**
 * Limpia un key de un Map (elimina si existe)
 */
export function clearFromMap<T>(
  map: Map<string, T>,
  key: string
): Map<string, T> {
  const newMap = new Map(map)
  newMap.delete(key)
  return newMap
}



