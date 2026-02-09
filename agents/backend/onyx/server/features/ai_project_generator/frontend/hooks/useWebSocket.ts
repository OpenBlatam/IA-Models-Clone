'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { API_CONFIG } from '@/lib/constants'
import type { WebSocketMessage } from '@/types'

export const useWebSocket = (projectId?: string) => {
  const [messages, setMessages] = useState<WebSocketMessage[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    const url = projectId
      ? `${API_CONFIG.WS_URL}/ws/project/${projectId}`
      : `${API_CONFIG.WS_URL}/ws`

    const ws = new WebSocket(url)

    ws.onopen = () => {
      setIsConnected(true)
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
    }

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data)
        setMessages((prev) => [...prev.slice(-49), message])
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    ws.onclose = () => {
      setIsConnected(false)
      reconnectTimeoutRef.current = setTimeout(() => {
        connect()
      }, 3000)
    }

    wsRef.current = ws
  }, [projectId])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
  }, [])

  const sendMessage = useCallback((message: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  return {
    messages,
    isConnected,
    sendMessage,
    connect,
    disconnect,
  }
}

