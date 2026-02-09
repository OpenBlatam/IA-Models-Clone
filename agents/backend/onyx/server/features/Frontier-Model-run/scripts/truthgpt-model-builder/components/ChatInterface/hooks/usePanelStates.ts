import { useState, useCallback } from 'react'

/**
 * Generic hook for managing panel states with message tracking
 * Eliminates duplicate code for panel state management
 */
export function usePanelStates() {
  const [panels, setPanels] = useState<Map<string, { visible: boolean; messages: Set<string> }>>(new Map())

  const togglePanel = useCallback((panelId: string) => {
    setPanels((prev) => {
      const newPanels = new Map(prev)
      const current = newPanels.get(panelId) || { visible: false, messages: new Set<string>() }
      newPanels.set(panelId, { ...current, visible: !current.visible })
      return newPanels
    })
  }, [])

  const setPanelVisible = useCallback((panelId: string, visible: boolean) => {
    setPanels((prev) => {
      const newPanels = new Map(prev)
      const current = newPanels.get(panelId) || { visible: false, messages: new Set<string>() }
      newPanels.set(panelId, { ...current, visible })
      return newPanels
    })
  }, [])

  const addMessageToPanel = useCallback((panelId: string, messageId: string) => {
    setPanels((prev) => {
      const newPanels = new Map(prev)
      const current = newPanels.get(panelId) || { visible: false, messages: new Set<string>() }
      const newMessages = new Set(current.messages)
      newMessages.add(messageId)
      newPanels.set(panelId, { ...current, messages: newMessages })
      return newPanels
    })
  }, [])

  const removeMessageFromPanel = useCallback((panelId: string, messageId: string) => {
    setPanels((prev) => {
      const newPanels = new Map(prev)
      const current = newPanels.get(panelId)
      if (current) {
        const newMessages = new Set(current.messages)
        newMessages.delete(messageId)
        newPanels.set(panelId, { ...current, messages: newMessages })
      }
      return newPanels
    })
  }, [])

  const clearPanelMessages = useCallback((panelId: string) => {
    setPanels((prev) => {
      const newPanels = new Map(prev)
      const current = newPanels.get(panelId)
      if (current) {
        newPanels.set(panelId, { ...current, messages: new Set() })
      }
      return newPanels
    })
  }, [])

  const isPanelVisible = useCallback((panelId: string) => {
    return panels.get(panelId)?.visible ?? false
  }, [panels])

  const getPanelMessages = useCallback((panelId: string) => {
    return panels.get(panelId)?.messages ?? new Set<string>()
  }, [panels])

  return {
    togglePanel,
    setPanelVisible,
    addMessageToPanel,
    removeMessageFromPanel,
    clearPanelMessages,
    isPanelVisible,
    getPanelMessages,
  }
}




