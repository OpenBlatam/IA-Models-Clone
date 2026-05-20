/**
 * Utilidades de Drag and Drop
 * ===========================
 * 
 * Funciones para manejo de drag and drop
 */

export interface DragDropConfig {
  onDragStart?: (event: DragEvent, data: any) => void
  onDragEnd?: (event: DragEvent) => void
  onDragOver?: (event: DragEvent) => void
  onDrop?: (event: DragEvent, data: any) => void
  dragData?: any
  dragType?: string
  effect?: 'move' | 'copy' | 'link' | 'none'
}

/**
 * Configura drag and drop en un elemento
 */
export function setupDragDrop(
  element: HTMLElement,
  config: DragDropConfig
): () => void {
  const {
    onDragStart,
    onDragEnd,
    onDragOver,
    onDrop,
    dragData,
    dragType = 'text/plain',
    effect = 'move'
  } = config

  const handleDragStart = (e: DragEvent) => {
    if (onDragStart) {
      onDragStart(e, dragData)
    }

    if (dragData !== undefined) {
      const data = typeof dragData === 'string' 
        ? dragData 
        : JSON.stringify(dragData)
      
      e.dataTransfer!.setData(dragType, data)
      e.dataTransfer!.effectAllowed = effect
    }
  }

  const handleDragEnd = (e: DragEvent) => {
    if (onDragEnd) {
      onDragEnd(e)
    }
  }

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault()
    e.dataTransfer!.dropEffect = effect
    
    if (onDragOver) {
      onDragOver(e)
    }
  }

  const handleDrop = (e: DragEvent) => {
    e.preventDefault()
    
    const data = e.dataTransfer!.getData(dragType)
    let parsedData = data

    try {
      parsedData = JSON.parse(data)
    } catch {
      // Si no es JSON, usar el string directamente
    }

    if (onDrop) {
      onDrop(e, parsedData)
    }
  }

  element.setAttribute('draggable', 'true')
  element.addEventListener('dragstart', handleDragStart)
  element.addEventListener('dragend', handleDragEnd)
  element.addEventListener('dragover', handleDragOver)
  element.addEventListener('drop', handleDrop)

  return () => {
    element.removeAttribute('draggable')
    element.removeEventListener('dragstart', handleDragStart)
    element.removeEventListener('dragend', handleDragEnd)
    element.removeEventListener('dragover', handleDragOver)
    element.removeEventListener('drop', handleDrop)
  }
}

/**
 * Obtiene datos de drag desde un evento
 */
export function getDragData(event: DragEvent, type: string = 'text/plain'): any {
  const data = event.dataTransfer!.getData(type)
  
  try {
    return JSON.parse(data)
  } catch {
    return data
  }
}

/**
 * Establece datos de drag en un evento
 */
export function setDragData(
  event: DragEvent,
  data: any,
  type: string = 'text/plain'
): void {
  const stringData = typeof data === 'string' 
    ? data 
    : JSON.stringify(data)
  
  event.dataTransfer!.setData(type, stringData)
}

/**
 * Verifica si un elemento es un drop target válido
 */
export function isValidDropTarget(
  event: DragEvent,
  allowedTypes: string[] = ['text/plain']
): boolean {
  const types = Array.from(event.dataTransfer!.types)
  return allowedTypes.some(type => types.includes(type))
}






