'use client';

import { useState } from 'react';
import { useMutation } from 'react-query';
import { apiClient } from '@/lib/api/client';
import { Platform, type ExtractProfileRequest, type GenerateContentRequest } from '@/types';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import Input from '@/components/UI/Input';
import Select from '@/components/UI/Select';
import Tabs from '@/components/UI/Tabs';
import { PLATFORM_OPTIONS } from '@/lib/constants';

const BatchPage = (): JSX.Element => {
  const [extractProfiles, setExtractProfiles] = useState<ExtractProfileRequest[]>([]);
  const [generateContent, setGenerateContent] = useState<GenerateContentRequest[]>([]);

  const extractMutation = useMutation(() =>
    apiClient.createBatchExtractProfiles(extractProfiles)
  );

  const generateMutation = useMutation(() =>
    apiClient.createBatchGenerateContent(generateContent)
  );

  const handleAddExtractProfile = (): void => {
    setExtractProfiles([...extractProfiles, { platform: Platform.INSTAGRAM, username: '' }]);
  };

  const handleRemoveExtractProfile = (index: number): void => {
    setExtractProfiles(extractProfiles.filter((_, i) => i !== index));
  };

  const handleExtractProfileChange = (
    index: number,
    field: keyof ExtractProfileRequest,
    value: string | boolean
  ): void => {
    const updated = [...extractProfiles];
    updated[index] = { ...updated[index], [field]: value };
    setExtractProfiles(updated);
  };

  const handleAddGenerateContent = (): void => {
    setGenerateContent([
      ...generateContent,
      {
        identity_profile_id: '',
        platform: Platform.INSTAGRAM,
        content_type: 'post' as const,
      },
    ]);
  };

  const handleRemoveGenerateContent = (index: number): void => {
    setGenerateContent(generateContent.filter((_, i) => i !== index));
  };

  const handleGenerateContentChange = (
    index: number,
    field: keyof GenerateContentRequest,
    value: string
  ): void => {
    const updated = [...generateContent];
    updated[index] = { ...updated[index], [field]: value };
    setGenerateContent(updated);
  };

  const handleSubmitExtract = (e: React.FormEvent<HTMLFormElement>): void => {
    e.preventDefault();
    if (extractProfiles.length === 0) {
      return;
    }
    extractMutation.mutate();
  };

  const handleSubmitGenerate = (e: React.FormEvent<HTMLFormElement>): void => {
    e.preventDefault();
    if (generateContent.length === 0) {
      return;
    }
    generateMutation.mutate();
  };

  const tabs = [
    {
      id: 'extract',
      label: 'Batch Extract Profiles',
      content: (
        <Card>
          <form onSubmit={handleSubmitExtract}>
            <div className="space-y-4">
              {extractProfiles.map((profile, index) => (
                <Card key={index} className="border">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="font-semibold">Profile {index + 1}</h3>
                      <Button
                        type="button"
                        variant="danger"
                        onClick={() => handleRemoveExtractProfile(index)}
                        className="text-sm"
                      >
                        Remove
                      </Button>
                    </div>
                    <Select
                      label="Platform"
                      value={profile.platform}
                      onChange={(e) =>
                        handleExtractProfileChange(index, 'platform', e.target.value as Platform)
                      }
                      options={PLATFORM_OPTIONS.map((opt) => ({
                        value: opt.value as Platform,
                        label: opt.label,
                      }))}
                    />
                    <Input
                      label="Username"
                      value={profile.username}
                      onChange={(e) => handleExtractProfileChange(index, 'username', e.target.value)}
                      required
                    />
                  </div>
                </Card>
              ))}
              <div className="flex gap-4">
                <Button type="button" variant="secondary" onClick={handleAddExtractProfile}>
                  Add Profile
                </Button>
                <Button type="submit" isLoading={extractMutation.isLoading} disabled={extractProfiles.length === 0}>
                  Submit Batch
                </Button>
              </div>
            </div>
          </form>
        </Card>
      ),
    },
    {
      id: 'generate',
      label: 'Batch Generate Content',
      content: (
        <Card>
          <form onSubmit={handleSubmitGenerate}>
            <div className="space-y-4">
              {generateContent.map((content, index) => (
                <Card key={index} className="border">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="font-semibold">Content {index + 1}</h3>
                      <Button
                        type="button"
                        variant="danger"
                        onClick={() => handleRemoveGenerateContent(index)}
                        className="text-sm"
                      >
                        Remove
                      </Button>
                    </div>
                    <Input
                      label="Identity ID"
                      value={content.identity_profile_id}
                      onChange={(e) =>
                        handleGenerateContentChange(index, 'identity_profile_id', e.target.value)
                      }
                      required
                    />
                    <Select
                      label="Platform"
                      value={content.platform}
                      onChange={(e) =>
                        handleGenerateContentChange(index, 'platform', e.target.value as Platform)
                      }
                      options={PLATFORM_OPTIONS.map((opt) => ({
                        value: opt.value as Platform,
                        label: opt.label,
                      }))}
                    />
                  </div>
                </Card>
              ))}
              <div className="flex gap-4">
                <Button type="button" variant="secondary" onClick={handleAddGenerateContent}>
                  Add Content
                </Button>
                <Button type="submit" isLoading={generateMutation.isLoading} disabled={generateContent.length === 0}>
                  Submit Batch
                </Button>
              </div>
            </div>
          </form>
        </Card>
      ),
    },
  ];

  return (
    <PageLayout>
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Batch Operations</h1>
        <Tabs tabs={tabs} />
      </div>
    </PageLayout>
  );
};

export default BatchPage;



