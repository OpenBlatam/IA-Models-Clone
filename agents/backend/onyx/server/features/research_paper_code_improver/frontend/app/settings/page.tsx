'use client'

import React from 'react'
import PageLayout from '@/components/layout/PageLayout'
import { Card, Button, Input, Select, Checkbox, Divider } from '@/components/ui'
import { HelpButton } from '@/components/features'
import { useLocalStorage } from '@/hooks'

interface Settings {
  theme: 'light' | 'dark' | 'auto'
  language: string
  notifications: boolean
  autoSave: boolean
  defaultModel: string
}

const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useLocalStorage<Settings>('settings', {
    theme: 'light',
    language: 'en',
    notifications: true,
    autoSave: true,
    defaultModel: '',
  })

  const handleSettingChange = <K extends keyof Settings>(
    key: K,
    value: Settings[K]
  ) => {
    setSettings((prev) => ({ ...prev, [key]: value }))
  }

  return (
    <PageLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
          <p className="mt-2 text-gray-600">
            Configure your preferences and application settings
          </p>
        </div>

        <Card>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">
              Appearance
            </h2>
            <HelpButton
              content={{
                title: 'Appearance Settings',
                sections: [
                  {
                    heading: 'Theme',
                    content:
                      'Choose your preferred color theme. Auto mode follows your system preferences.',
                  },
                ],
              }}
            />
          </div>
          <div className="space-y-4">
            <Select
              label="Theme"
              value={settings.theme}
              onChange={(e) =>
                handleSettingChange('theme', e.target.value as Settings['theme'])
              }
              options={[
                { value: 'light', label: 'Light' },
                { value: 'dark', label: 'Dark' },
                { value: 'auto', label: 'Auto (System)' },
              ]}
            />
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">General</h2>
            <HelpButton
              content={{
                title: 'General Settings',
                sections: [
                  {
                    heading: 'Notifications',
                    content:
                      'Enable or disable browser notifications for important events.',
                  },
                  {
                    heading: 'Auto Save',
                    content:
                      'Automatically save your work and preferences.',
                  },
                ],
              }}
            />
          </div>
          <div className="space-y-4">
            <Checkbox
              label="Enable Notifications"
              checked={settings.notifications}
              onChange={(e) =>
                handleSettingChange('notifications', e.target.checked)
              }
            />
            <Checkbox
              label="Auto Save"
              checked={settings.autoSave}
              onChange={(e) =>
                handleSettingChange('autoSave', e.target.checked)
              }
            />
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">
              Code Improvement
            </h2>
            <HelpButton
              content={{
                title: 'Code Improvement Settings',
                sections: [
                  {
                    heading: 'Default Model',
                    content:
                      'Select the default AI model to use for code improvements. Leave empty to use the system default.',
                  },
                ],
              }}
            />
          </div>
          <div className="space-y-4">
            <Input
              label="Default Model ID"
              placeholder="Leave empty for system default"
              value={settings.defaultModel}
              onChange={(e) =>
                handleSettingChange('defaultModel', e.target.value)
              }
              helperText="The model ID to use by default for code improvements"
            />
          </div>
        </Card>

        <div className="flex justify-end gap-4">
          <Button variant="outline" onClick={() => window.location.reload()}>
            Reset to Defaults
          </Button>
          <Button
            variant="primary"
            onClick={() => {
              // Settings are auto-saved via useLocalStorage
              alert('Settings saved!')
            }}
          >
            Save Settings
          </Button>
        </div>
      </div>
    </PageLayout>
  )
}

export default SettingsPage




