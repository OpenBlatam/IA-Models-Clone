'use client';

import { useState } from 'react';
import { useMutation } from 'react-query';
import { apiClient } from '@/lib/api/client';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import Input from '@/components/UI/Input';
import LoadingSpinner from '@/components/UI/LoadingSpinner';

const WEBHOOK_EVENTS = [
  { value: 'identity_created', label: 'Identity Created' },
  { value: 'content_generated', label: 'Content Generated' },
  { value: 'task_completed', label: 'Task Completed' },
  { value: 'task_failed', label: 'Task Failed' },
];

const WebhooksPage = (): JSX.Element => {
  const [formData, setFormData] = useState({
    url: '',
    events: [] as string[],
  });

  const registerMutation = useMutation(() => apiClient.registerWebhook(formData.url, formData.events));

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>): void => {
    e.preventDefault();
    registerMutation.mutate();
  };

  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setFormData({ ...formData, url: e.target.value });
  };

  const handleEventToggle = (eventValue: string): void => {
    const events = formData.events.includes(eventValue)
      ? formData.events.filter((e) => e !== eventValue)
      : [...formData.events, eventValue];
    setFormData({ ...formData, events });
  };

  return (
    <PageLayout>
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Webhooks</h1>
        <Card title="Register Webhook">
          <form onSubmit={handleSubmit}>
            <div className="space-y-4">
              <Input
                label="Webhook URL"
                type="url"
                value={formData.url}
                onChange={handleUrlChange}
                placeholder="https://your-app.com/webhook"
                required
              />
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Events</label>
                <div className="space-y-2">
                  {WEBHOOK_EVENTS.map((event) => (
                    <label key={event.value} className="flex items-center">
                      <input
                        type="checkbox"
                        checked={formData.events.includes(event.value)}
                        onChange={() => handleEventToggle(event.value)}
                        className="mr-2"
                        aria-label={event.label}
                      />
                      <span>{event.label}</span>
                    </label>
                  ))}
                </div>
              </div>
              <Button type="submit" isLoading={registerMutation.isLoading} className="w-full">
                Register Webhook
              </Button>
            </div>
          </form>
        </Card>
      </div>
    </PageLayout>
  );
};

export default WebhooksPage;

