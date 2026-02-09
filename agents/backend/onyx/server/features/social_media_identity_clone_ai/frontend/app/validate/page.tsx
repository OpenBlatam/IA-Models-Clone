'use client';

import { useState } from 'react';
import { useMutation } from 'react-query';
import { apiClient } from '@/lib/api/client';
import type { ContentValidation } from '@/types';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import Input from '@/components/UI/Input';
import Textarea from '@/components/UI/Textarea';
import Tabs from '@/components/UI/Tabs';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import StatusBadge from '@/components/UI/StatusBadge';

const ValidatePage = (): JSX.Element => {
  const [contentId, setContentId] = useState('');
  const [directContent, setDirectContent] = useState('');
  const [validationResult, setValidationResult] = useState<ContentValidation | null>(null);

  const validateByIdMutation = useMutation(
    (id: string) => apiClient.validateContent(id),
    {
      onSuccess: (data) => {
        setValidationResult(data);
      },
    }
  );

  const validateDirectMutation = useMutation(
    (content: string) => apiClient.validateContentDirect(content),
    {
      onSuccess: (data) => {
        setValidationResult(data);
      },
    }
  );

  const handleContentIdChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setContentId(e.target.value);
  };

  const handleDirectContentChange = (e: React.ChangeEvent<HTMLTextAreaElement>): void => {
    setDirectContent(e.target.value);
  };

  const handleValidateById = (e: React.FormEvent<HTMLFormElement>): void => {
    e.preventDefault();
    if (!contentId) {
      return;
    }
    validateByIdMutation.mutate(contentId);
  };

  const handleValidateDirect = (e: React.FormEvent<HTMLFormElement>): void => {
    e.preventDefault();
    if (!directContent.trim()) {
      return;
    }
    validateDirectMutation.mutate(directContent);
  };

  const tabs = [
    {
      id: 'by-id',
      label: 'Validate by Content ID',
      content: (
        <Card>
          <form onSubmit={handleValidateById}>
            <div className="space-y-4">
              <Input
                label="Content ID"
                value={contentId}
                onChange={handleContentIdChange}
                placeholder="Enter content ID to validate"
                required
              />
              <Button
                type="submit"
                isLoading={validateByIdMutation.isLoading}
                className="w-full"
              >
                Validate
              </Button>
            </div>
          </form>
        </Card>
      ),
    },
    {
      id: 'direct',
      label: 'Validate Direct Content',
      content: (
        <Card>
          <form onSubmit={handleValidateDirect}>
            <div className="space-y-4">
              <Textarea
                label="Content"
                value={directContent}
                onChange={handleDirectContentChange}
                placeholder="Enter content to validate"
                required
                className="min-h-[200px]"
              />
              <Button
                type="submit"
                isLoading={validateDirectMutation.isLoading}
                className="w-full"
              >
                Validate
              </Button>
            </div>
          </form>
        </Card>
      ),
    },
  ];

  return (
    <PageLayout>
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Content Validation</h1>
        <Tabs tabs={tabs} />
        {validationResult && (
          <Card title="Validation Result" className="mt-8">
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <span className="font-semibold">Status:</span>
                <StatusBadge
                  status={validationResult.is_valid ? 'Valid' : 'Invalid'}
                  variant={validationResult.is_valid ? 'success' : 'error'}
                />
              </div>
              {validationResult.errors && validationResult.errors.length > 0 && (
                <div>
                  <h3 className="font-semibold mb-2">Errors:</h3>
                  <ul className="list-disc list-inside space-y-1">
                    {validationResult.errors.map((error, idx) => (
                      <li key={idx} className="text-red-600">
                        {error}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {validationResult.warnings && validationResult.warnings.length > 0 && (
                <div>
                  <h3 className="font-semibold mb-2">Warnings:</h3>
                  <ul className="list-disc list-inside space-y-1">
                    {validationResult.warnings.map((warning, idx) => (
                      <li key={idx} className="text-yellow-600">
                        {warning}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {validationResult.suggestions && validationResult.suggestions.length > 0 && (
                <div>
                  <h3 className="font-semibold mb-2">Suggestions:</h3>
                  <ul className="list-disc list-inside space-y-1">
                    {validationResult.suggestions.map((suggestion, idx) => (
                      <li key={idx} className="text-blue-600">
                        {suggestion}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </Card>
        )}
      </div>
    </PageLayout>
  );
};

export default ValidatePage;



