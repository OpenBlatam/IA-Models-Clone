/**
 * Hook para herramientas de desarrollo y debugging
 * ================================================
 */

import { useState, useCallback, useRef } from 'react'

export interface DevToolsState {
  enabled: boolean
  verbose: boolean
  showMetrics: boolean
  showTimings: boolean
  logLevel: 'none' | 'error' | 'warn' | 'info' | 'debug'
}

export interface UseModelDevToolsResult {
  state: DevToolsState
  toggle: () => void
  setVerbose: (verbose: boolean) => void
  setLogLevel: (level: DevToolsState['logLevel']) => void
  log: (level: 'error' | 'warn' | 'info' | 'debug', message: string, data?: any) => void
  time: (label: string) => () => void
  measure: <T>(label: string, fn: () => Promise<T>) => Promise<T>
  getMetrics: () => {
    logs: Array<{ level: string; message: string; timestamp: number; data?: any }>
    timings: Array<{ label: string; duration: number; timestamp: number }>
  }
  clear: () => void
}

/**
 * Hook para herramientas de desarrollo
 */
export function useModelDevTools(
  initialState: Partial<DevToolsState> = {}
): UseModelDevToolsResult {
  const [state, setState] = useState<DevToolsState>({
    enabled: process.env.NODE_ENV === 'development' || initialState.enabled || false,
    verbose: initialState.verbose || false,
    showMetrics: initialState.showMetrics || false,
    showTimings: initialState.showTimings || false,
    logLevel: initialState.logLevel || 'info'
  })

  const logsRef = useRef<Array<{ level: string; message: string; timestamp: number; data?: any }>>([])
  const timingsRef = useRef<Array<{ label: string; duration: number; timestamp: number }>>([])
  const timersRef = useRef<Map<string, number>>(new Map())

  const logLevels = {
    none: 0,
    error: 1,
    warn: 2,
    info: 3,
    debug: 4
  }

  const log = useCallback((
    level: 'error' | 'warn' | 'info' | 'debug',
    message: string,
    data?: any
  ) => {
    if (!state.enabled || logLevels[level] > logLevels[state.logLevel]) {
      return
    }

    const entry = {
      level,
      message,
      timestamp: Date.now(),
      data
    }

    logsRef.current.push(entry)
    
    // Mantener solo los últimos 1000 logs
    if (logsRef.current.length > 1000) {
      logsRef.current.shift()
    }

    // Log a consola si está habilitado
    if (state.verbose) {
      const consoleMethod = console[level] || console.log
      consoleMethod(`[ModelDevTools] ${level.toUpperCase()}:`, message, data || '')
    }

    // Log nativo según nivel
    switch (level) {
      case 'error':
        console.error(`[ModelDevTools]`, message, data || '')
        break
      case 'warn':
        console.warn(`[ModelDevTools]`, message, data || '')
        break
      case 'info':
        if (state.verbose) {
          console.info(`[ModelDevTools]`, message, data || '')
        }
        break
      case 'debug':
        if (state.verbose) {
          console.debug(`[ModelDevTools]`, message, data || '')
        }
        break
    }
  }, [state.enabled, state.verbose, state.logLevel])

  const time = useCallback((label: string) => {
    const start = Date.now()
    timersRef.current.set(label, start)

    return () => {
      const end = Date.now()
      const duration = end - start
      timersRef.current.delete(label)

      timingsRef.current.push({
        label,
        duration,
        timestamp: end
      })

      // Mantener solo los últimos 500 timings
      if (timingsRef.current.length > 500) {
        timingsRef.current.shift()
      }

      if (state.showTimings) {
        log('debug', `Timing: ${label}`, `${duration}ms`)
      }
    }
  }, [state.showTimings, log])

  const measure = useCallback(async <T,>(
    label: string,
    fn: () => Promise<T>
  ): Promise<T> => {
    const stopTimer = time(label)
    try {
      const result = await fn()
      stopTimer()
      return result
    } catch (error) {
      stopTimer()
      log('error', `Error in ${label}:`, error)
      throw error
    }
  }, [time, log])

  const toggle = useCallback(() => {
    setState(prev => ({ ...prev, enabled: !prev.enabled }))
  }, [])

  const setVerbose = useCallback((verbose: boolean) => {
    setState(prev => ({ ...prev, verbose }))
  }, [])

  const setLogLevel = useCallback((level: DevToolsState['logLevel']) => {
    setState(prev => ({ ...prev, logLevel: level }))
  }, [])

  const getMetrics = useCallback(() => {
    return {
      logs: [...logsRef.current],
      timings: [...timingsRef.current]
    }
  }, [])

  const clear = useCallback(() => {
    logsRef.current = []
    timingsRef.current = []
    timersRef.current.clear()
  }, [])

  return {
    state,
    toggle,
    setVerbose,
    setLogLevel,
    log,
    time,
    measure,
    getMetrics,
    clear
  }
}

