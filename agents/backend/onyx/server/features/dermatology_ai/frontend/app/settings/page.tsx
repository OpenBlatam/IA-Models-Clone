'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Settings, Bell, Shield, Globe } from 'lucide-react';
import { toastMessages, showSuccess } from '@/lib/utils/toastUtils';
import { PageLayout } from '@/components/layout/PageLayout';
import { PageHeader } from '@/components/layout/PageHeader';
import { Toggle } from '@/components/ui/Toggle';
import { FormSection } from '@/components/ui/FormSection';

export default function SettingsPage() {
  const [apiUrl, setApiUrl] = useState(
    typeof window !== 'undefined'
      ? localStorage.getItem('api_url') || 'http://localhost:8006'
      : 'http://localhost:8006'
  );
  const [notifications, setNotifications] = useState({
    email: true,
    push: false,
    alerts: true,
  });

  const handleSaveApiUrl = () => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('api_url', apiUrl);
      showSuccess(toastMessages.settingsSaved);
    }
  };

  const handleSaveNotifications = () => {
    showSuccess(toastMessages.settingsSaved);
  };

  return (
    <PageLayout maxWidth="4xl">
      <PageHeader
        title="Settings"
        description="Customize your experience and preferences"
        icon={Settings}
      />

        <div className="space-y-6">
          {/* API Configuration */}
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-2">
                <Globe className="h-5 w-5 text-primary-600" />
                <CardTitle>API Configuration</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <FormSection spacing={4}>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Backend URL
                  </label>
                  <input
                    type="text"
                    value={apiUrl}
                    onChange={(e) => setApiUrl(e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 dark:border-gray-700 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent dark:bg-gray-800 dark:text-white"
                    placeholder="https://api.example.com"
                  />
                  <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                    API server base URL
                  </p>
                </div>
                <Button onClick={handleSaveApiUrl}>Save</Button>
              </FormSection>
            </CardContent>
          </Card>

          {/* Notifications */}
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-2">
                <Bell className="h-5 w-5 text-primary-600" />
                <CardTitle>Notifications</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <FormSection spacing={4}>
                <Toggle
                  checked={notifications.email}
                  onChange={(checked) =>
                    setNotifications({ ...notifications, email: checked })
                  }
                  label="Email Notifications"
                  description="Get updates via email"
                />
                <Toggle
                  checked={notifications.push}
                  onChange={(checked) =>
                    setNotifications({ ...notifications, push: checked })
                  }
                  label="Push Notifications"
                  description="Get real-time notifications"
                />
                <Toggle
                  checked={notifications.alerts}
                  onChange={(checked) =>
                    setNotifications({ ...notifications, alerts: checked })
                  }
                  label="Important Alerts"
                  description="Get alerts on conditions"
                />
                <Button onClick={handleSaveNotifications}>Save Preferences</Button>
              </FormSection>
            </CardContent>
          </Card>

          {/* About */}
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-2">
                <Shield className="h-5 w-5 text-primary-600" />
                <CardTitle>About</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>Version:</strong> 1.0.0
                </p>
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>Backend API:</strong> Dermatology AI v5.5.0
                </p>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-4">
                  AI-powered skin analysis and personalized skincare recommendations
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
    </PageLayout>
  );
}

