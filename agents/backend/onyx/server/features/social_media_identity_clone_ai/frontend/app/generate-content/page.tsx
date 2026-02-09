'use client';

import { useState } from 'react';
import { Platform, ContentType, type GenerateContentRequest } from '@/types';
import { useGenerateContentMutation } from '@/lib/modules/api';
import { getPlatformOptions } from '@/lib/modules/platform';
import { getContentTypeOptions } from '@/lib/modules/content';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import Input from '@/components/UI/Input';
import Select from '@/components/UI/Select';
import CopyButton from '@/components/UI/CopyButton';
import MutationError from '@/components/Forms/MutationError';
import MutationLoading from '@/components/Forms/MutationLoading';
import TagsList from '@/components/Display/TagsList';
import InfoRow from '@/components/Display/InfoRow';

const GenerateContentPage = (): JSX.Element => {
  const [identityId, setIdentityId] = useState('');
  const [platform, setPlatform] = useState<Platform>(Platform.INSTAGRAM);
  const [contentType, setContentType] = useState<ContentType>(ContentType.POST);
  const [topic, setTopic] = useState('');
  const [style, setStyle] = useState('');
  const [duration, setDuration] = useState(60);
  const [videoTitle, setVideoTitle] = useState('');

  const generateMutation = useGenerateContentMutation();

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>): void => {
    e.preventDefault();
    
    const trimmedIdentityId = identityId.trim();
    if (!trimmedIdentityId) {
      return;
    }

    const request: GenerateContentRequest = {
      identity_profile_id: trimmedIdentityId,
      platform,
      content_type: contentType,
    };

    const trimmedTopic = topic.trim();
    const trimmedStyle = style.trim();
    const trimmedVideoTitle = videoTitle.trim();

    if (trimmedTopic) {
      request.topic = trimmedTopic;
    }
    if (trimmedStyle) {
      request.style = trimmedStyle;
    }
    if (contentType === ContentType.VIDEO && duration) {
      request.duration = duration;
    }
    if (platform === Platform.YOUTUBE && trimmedVideoTitle) {
      request.video_title = trimmedVideoTitle;
    }

    generateMutation.mutate(request);
  };

  const handleIdentityChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setIdentityId(e.target.value);
  };

  const handlePlatformChange = (e: React.ChangeEvent<HTMLSelectElement>): void => {
    setPlatform(e.target.value as Platform);
  };

  const handleContentTypeChange = (e: React.ChangeEvent<HTMLSelectElement>): void => {
    setContentType(e.target.value as ContentType);
  };

  const handleTopicChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setTopic(e.target.value);
  };

  const handleStyleChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setStyle(e.target.value);
  };

  const handleDurationChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    const value = Number(e.target.value);
    if (!isNaN(value) && value > 0) {
      setDuration(value);
    }
  };

  const handleVideoTitleChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setVideoTitle(e.target.value);
  };

  return (
    <PageLayout>
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Generate Content</h1>

        <Card title="Generate Content Based on Identity">
          <form onSubmit={handleSubmit}>
            <div className="space-y-4">
              <Input
                label="Identity ID"
                type="text"
                value={identityId}
                onChange={handleIdentityChange}
                placeholder="Enter identity profile ID"
                required
              />

              <Select
                label="Platform"
                value={platform}
                onChange={handlePlatformChange}
                options={getPlatformOptions()}
              />

              <Select
                label="Content Type"
                value={contentType}
                onChange={handleContentTypeChange}
                options={getContentTypeOptions()}
              />

              <Input
                label="Topic (optional)"
                type="text"
                value={topic}
                onChange={handleTopicChange}
                placeholder="e.g., fitness, cooking, travel"
              />

              <Input
                label="Style (optional)"
                type="text"
                value={style}
                onChange={handleStyleChange}
                placeholder="e.g., motivational, casual, professional"
              />

              {contentType === ContentType.VIDEO && (
                <Input
                  label="Duration (seconds)"
                  type="number"
                  value={duration}
                  onChange={handleDurationChange}
                  min={1}
                  max={600}
                />
              )}

              {platform === Platform.YOUTUBE && (
                <Input
                  label="Video Title (optional)"
                  type="text"
                  value={videoTitle}
                  onChange={handleVideoTitleChange}
                  placeholder="Enter video title"
                />
              )}

              <Button type="submit" isLoading={generateMutation.isLoading} className="w-full">
                Generate Content
              </Button>
            </div>
          </form>
        </Card>

        <MutationLoading isLoading={generateMutation.isLoading} message="Generating content..." />
        <MutationError error={generateMutation.error} />

        {generateMutation.data && (
          <Card className="mt-6" title="Generated Content">
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-600 mb-2">Content ID</p>
                <div className="flex items-center gap-2">
                  <p className="font-mono text-sm">{generateMutation.data.content_id}</p>
                  <CopyButton text={generateMutation.data.content_id} label="Copy" variant="secondary" className="text-xs" />
                </div>
              </div>

              {generateMutation.data.content.title && (
                <InfoRow
                  label="Title"
                  value={generateMutation.data.content.title}
                  valueClassName="font-semibold text-lg"
                />
              )}

              <div>
                <p className="text-sm text-gray-600 mb-2">Content</p>
                <div className="bg-gray-50 p-4 rounded-lg whitespace-pre-wrap">
                  {generateMutation.data.content.content}
                </div>
              </div>

              {generateMutation.data.content.hashtags.length > 0 && (
                <div>
                  <p className="text-sm text-gray-600 mb-2">Hashtags</p>
                  <TagsList tags={generateMutation.data.content.hashtags} variant="primary" />
                </div>
              )}

              {generateMutation.data.validation && (
                <div className="pt-4 border-t">
                  <p className="text-sm text-gray-600 mb-2">Validation</p>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span>Score</span>
                      <span className="font-semibold">
                        {(generateMutation.data.validation.score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex items-center">
                      <span>Valid</span>
                      <span
                        className={cn(
                          'ml-2 px-2 py-1 rounded text-sm',
                          generateMutation.data.validation.is_valid
                            ? 'bg-green-100 text-green-700'
                            : 'bg-red-100 text-red-700'
                        )}
                      >
                        {generateMutation.data.validation.is_valid ? 'Yes' : 'No'}
                      </span>
                    </div>
                    {generateMutation.data.validation.suggestions.length > 0 && (
                      <div>
                        <p className="text-sm font-semibold mb-1">Suggestions:</p>
                        <ul className="list-disc list-inside text-sm text-gray-600">
                          {generateMutation.data.validation.suggestions.map((suggestion, idx) => (
                            <li key={idx}>{suggestion}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </Card>
        )}
      </div>
    </PageLayout>
  );
};

export default GenerateContentPage;

