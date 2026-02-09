'use client';

import { useQuery } from 'react-query';
import { useParams, useRouter } from 'next/navigation';
import { apiClient } from '@/lib/api/client';
import type { Template } from '@/types';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import { formatDate } from '@/lib/utils';
import PlatformBadge from '@/components/UI/PlatformBadge';

const TemplateDetailPage = (): JSX.Element => {
  const params = useParams();
  const router = useRouter();
  const templateId = params.id as string;

  const { data: template, isLoading } = useQuery<Template>(
    ['template', templateId],
    () => apiClient.getTemplate(templateId),
    { enabled: !!templateId }
  );

  const handleBack = (): void => {
    router.push('/templates');
  };

  if (isLoading) {
    return (
      <PageLayout>
        <LoadingSpinner />
      </PageLayout>
    );
  }

  if (!template) {
    return (
      <PageLayout>
        <Card>
          <p className="text-center text-gray-600 py-8">Template not found</p>
          <div className="text-center">
            <Button onClick={handleBack}>Back to Templates</Button>
          </div>
        </Card>
      </PageLayout>
    );
  }

  return (
    <PageLayout>
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">Template Details</h1>
          <Button variant="secondary" onClick={handleBack}>
            Back
          </Button>
        </div>

        <Card>
          <div className="space-y-4">
            <div>
              <h2 className="text-2xl font-semibold mb-2">{template.name}</h2>
              <div className="flex items-center gap-2 mb-4">
                <PlatformBadge platform={template.platform} />
                <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-sm">
                  {template.content_type}
                </span>
              </div>
            </div>

            {template.description && (
              <div>
                <p className="text-sm text-gray-600 mb-1">Description</p>
                <p className="text-gray-900">{template.description}</p>
              </div>
            )}

            <div>
              <p className="text-sm text-gray-600 mb-1">Template Content</p>
              <div className="bg-gray-50 p-4 rounded-lg">
                <pre className="whitespace-pre-wrap text-sm">{template.template}</pre>
              </div>
            </div>

            <div className="flex items-center justify-between pt-4 border-t">
              <div>
                <p className="text-xs text-gray-500">Created</p>
                <p className="text-sm">{formatDate(template.created_at)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Template ID</p>
                <p className="text-sm font-mono">{template.template_id}</p>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </PageLayout>
  );
};

export default TemplateDetailPage;



