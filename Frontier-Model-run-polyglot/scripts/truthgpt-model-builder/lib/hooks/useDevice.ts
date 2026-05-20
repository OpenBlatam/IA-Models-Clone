/**
 * Hook useDevice
 * ==============
 * 
 * Hook para detectar información del dispositivo
 */

import { useState, useEffect } from 'react'
import { getDeviceInfo, getDeviceType, getOS, getBrowser, isTouchDevice, prefersDarkMode, prefersReducedMotion } from '../utils/deviceUtils'

export interface UseDeviceResult {
  type: 'mobile' | 'tablet' | 'desktop'
  os: 'windows' | 'macos' | 'linux' | 'android' | 'ios' | 'unknown'
  browser: 'chrome' | 'firefox' | 'safari' | 'edge' | 'opera' | 'unknown'
  isTouch: boolean
  prefersDark: boolean
  prefersReducedMotion: boolean
  info: ReturnType<typeof getDeviceInfo>
}

/**
 * Hook para detectar información del dispositivo
 */
export function useDevice(): UseDeviceResult {
  const [deviceInfo, setDeviceInfo] = useState(() => getDeviceInfo())

  useEffect(() => {
    const handleResize = () => {
      setDeviceInfo(getDeviceInfo())
    }

    const handleDarkModeChange = () => {
      setDeviceInfo(getDeviceInfo())
    }

    const handleReducedMotionChange = () => {
      setDeviceInfo(getDeviceInfo())
    }

    window.addEventListener('resize', handleResize)
    
    const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    
    darkModeQuery.addEventListener('change', handleDarkModeChange)
    reducedMotionQuery.addEventListener('change', handleReducedMotionChange)

    return () => {
      window.removeEventListener('resize', handleResize)
      darkModeQuery.removeEventListener('change', handleDarkModeChange)
      reducedMotionQuery.removeEventListener('change', handleReducedMotionChange)
    }
  }, [])

  return {
    type: deviceInfo.type,
    os: deviceInfo.os,
    browser: deviceInfo.browser,
    isTouch: deviceInfo.isTouch,
    prefersDark: deviceInfo.prefersDark,
    prefersReducedMotion: deviceInfo.prefersReducedMotion,
    info: deviceInfo
  }
}







