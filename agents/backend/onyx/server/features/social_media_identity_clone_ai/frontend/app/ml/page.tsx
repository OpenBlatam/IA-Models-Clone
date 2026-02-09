'use client';

import { useState } from 'react';
import { useMutation, useQuery } from 'react-query';
import { apiClient } from '@/lib/api/client';
import { Platform } from '@/types';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import Input from '@/components/UI/Input';
import Select from '@/components/UI/Select';
import Textarea from '@/components/UI/Textarea';
import Tabs from '@/components/UI/Tabs';
import { PLATFORM_OPTIONS } from '@/lib/constants';

const MLPage = (): JSX.Element => {
  const [predictForm, setPredictForm] = useState({
    content: '',
    platform: Platform.INSTAGRAM,
    identity_id: '',
  });
  const [analyzeIdentityId, setAnalyzeIdentityId] = useState('');

  const predictMutation = useMutation(
    () => apiClient.predictPerformance(predictForm.content, predictForm.platform, predictForm.identity_id)
  );

  const { data: trends, isLoading: trendsLoading } = useQuery(
    ['analyze-trends', analyzeIdentityId],
    () => apiClient.analyzeTrends(analyzeIdentityId),
    { enabled: analyzeIdentityId.length > 0 }
  );

  const handlePredictSubmit = (e: React.FormEvent<HTMLFormElement>): void => {
    e.preventDefault();
    predictMutation.mutate();
  };

  const handleContentChange = (e: React.ChangeEvent<HTMLTextAreaElement>): void => {
    setPredictForm({ ...predictForm, content: e.target.value });
  };

  const handlePlatformChange = (e: React.ChangeEvent<HTMLSelectElement>): void => {
    setPredictForm({ ...predictForm, platform: e.target.value as Platform });
  };

  const handleIdentityIdChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setPredictForm({ ...predictForm, identity_id: e.target.value });
  };

  const handleAnalyzeIdentityIdChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setAnalyzeIdentityId(e.target.value);
  };

  const tabs = [
    {
      id: 'predict',
      label: 'Predict Performance',
      content: (
        <Card>
          <form onSubmit={handlePredictSubmit}>
            <div className="space-y-4">
              <Input
                label="Identity ID"
                value={predictForm.identity_id}
                onChange={handleIdentityIdChange}
                required
              />
              <Select
                label="Platform"
                value={predictForm.platform}
                onChange={handlePlatformChange}
                options={PLATFORM_OPTIONS.map((opt) => ({
                  value: opt.value as Platform,
                  label: opt.label,
                }))}
              />
              <Textarea
                label="Content"
                value={predictForm.content}
                onChange={handleContentChange}
                required
                className="min-h-[200px]"
              />
              <Button type="submit" isLoading={predictMutation.isLoading} className="w-full">
                Predict Performance
              </Button>
            </div>
          </form>
          {predictMutation.data && (
            <Card className="mt-6">
              <pre className="text-sm overflow-auto">
                {JSON.stringify(predictMutation.data, null, 2)}
              </pre>
            </Card>
          )}
        </Card>
      ),
    },
    {
      id: 'trends',
      label: 'Analyze Trends',
      content: (
        <Card>
          <div className="space-y-4">
            <Input
              label="Identity ID"
              value={analyzeIdentityId}
              onChange={handleAnalyzeIdentityIdChange}
              placeholder="Enter identity ID to analyze trends"
            />
            {trendsLoading ? (
              <div className="text-center py-8">Loading trends...</div>
            ) : trends ? (
              <Card>
                <pre className="text-sm overflow-auto">
                  {JSON.stringify(trends, null, 2)}
                </pre>
              </Card>
            ) : null}
          </div>
        </Card>
      ),
    },
  ];

  return (
    <PageLayout>
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Machine Learning</h1>
        <Tabs tabs={tabs} />
      </div>
    </PageLayout>
  );
};

export default MLPage;



