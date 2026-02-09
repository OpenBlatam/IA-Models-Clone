'use client'

import React, { useState, useEffect } from 'react'
import { Sun, Moon, Monitor } from 'lucide-react'
import { Button, Dropdown } from '../ui'
import { useLocalStorage } from '@/hooks'

type Theme = 'light' | 'dark' | 'system'

const ThemeSwitcher: React.FC = () => {
  const [theme, setTheme] = useLocalStorage<Theme>('theme', 'system')
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    applyTheme(theme)
  }, [theme])

  const applyTheme = (newTheme: Theme) => {
    const root = window.document.documentElement
    root.classList.remove('light', 'dark')

    if (newTheme === 'system') {
      const systemTheme = window.matchMedia('(prefers-color-scheme: dark)')
        .matches
        ? 'dark'
        : 'light'
      root.classList.add(systemTheme)
    } else {
      root.classList.add(newTheme)
    }
  }

  const handleThemeChange = (newTheme: Theme) => {
    setTheme(newTheme)
    applyTheme(newTheme)
  }

  if (!mounted) {
    return (
      <Button variant="ghost" size="sm" disabled>
        <Sun className="w-4 h-4" />
      </Button>
    )
  }

  const themeOptions = [
    {
      value: 'light',
      label: 'Light',
      icon: <Sun className="w-4 h-4" />,
    },
    {
      value: 'dark',
      label: 'Dark',
      icon: <Moon className="w-4 h-4" />,
    },
    {
      value: 'system',
      label: 'System',
      icon: <Monitor className="w-4 h-4" />,
    },
  ]

  const currentThemeOption = themeOptions.find((opt) => opt.value === theme)

  return (
    <Dropdown
      trigger={
        <Button variant="ghost" size="sm" title="Toggle theme">
          {currentThemeOption?.icon}
        </Button>
      }
      items={themeOptions.map((option) => ({
        label: option.label,
        icon: option.icon,
        onClick: () => handleThemeChange(option.value as Theme),
        active: theme === option.value,
      }))}
    />
  )
}

export default ThemeSwitcher



