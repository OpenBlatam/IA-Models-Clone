'use client';

import { useState } from 'react';
import type { BuildIdentityResponse } from '@/types';
import { useIdentityForm } from '@/lib/features/identities';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import Input from '@/components/UI/Input';
import MutationError from '@/components/Forms/MutationError';
import MutationLoading from '@/components/Forms/MutationLoading';
import StatsGrid from '@/components/Display/StatsGrid';
import TagsList from '@/components/Display/TagsList';

const BuildIdentityPage = (): JSX.Element => {
  const [result, setResult] = useState<BuildIdentityResponse | null>(null);
  const {
    values,
    errors,
    handleChange,
    handleSubmit,
    isSubmitting,
  } = useIdentityForm();

  const handleFormSubmit = async (e: React.FormEvent<HTMLFormElement>): Promise<void> => {
    const data = await handleSubmit(e);
    if (data) {
      setResult(data);
    }
  };

  const handleViewIdentity = (): void => {
    if (!result) {
      return;
    }
    // Navigation is handled by useIdentityOperations
  };

  return (
    <PageLayout>
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Build Identity Profile</h1>

        <Card title="Build Identity from Social Media Profiles">
          <form onSubmit={handleFormSubmit}>
            <div className="space-y-4">
              <Input
                label="TikTok Username (optional)"
                type="text"
                value={values.tiktokUsername}
                onChange={handleChange('tiktokUsername')}
                placeholder="Enter TikTok username"
                error={errors.tiktokUsername}
              />

              <Input
                label="Instagram Username (optional)"
                type="text"
                value={values.instagramUsername}
                onChange={handleChange('instagramUsername')}
                placeholder="Enter Instagram username"
                error={errors.instagramUsername}
              />

              <Input
                label="YouTube Channel ID (optional)"
                type="text"
                value={values.youtubeChannelId}
                onChange={handleChange('youtubeChannelId')}
                placeholder="Enter YouTube channel ID"
                error={errors.youtubeChannelId}
              />

              <p className="text-sm text-gray-600">
                Provide at least one profile to build the identity
              </p>

              <Button type="submit" isLoading={isSubmitting} className="w-full">
                Build Identity
              </Button>
            </div>
          </form>
        </Card>

        <MutationLoading
          isLoading={isSubmitting}
          message="This may take a few minutes. Please wait..."
        />

        {result && (
          <Card className="mt-6" title="Identity Built Successfully">
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-600">Identity ID</p>
                <p className="font-mono text-sm">{result.identity_id}</p>
              </div>

              <div>
                <p className="text-sm text-gray-600">Username</p>
                <p className="font-semibold text-lg">{result.identity.username}</p>
              </div>

              {result.identity.display_name && (
                <div>
                  <p className="text-sm text-gray-600">Display Name</p>
                  <p className="font-semibold">{result.identity.display_name}</p>
                </div>
              )}

              {result.identity.bio && (
                <div>
                  <p className="text-sm text-gray-600">Bio</p>
                  <p>{result.identity.bio}</p>
                </div>
              )}

              <StatsGrid
                stats={[
                  { label: 'Videos', value: result.stats.total_videos },
                  { label: 'Posts', value: result.stats.total_posts },
                  { label: 'Comments', value: result.stats.total_comments },
                  { label: 'Topics', value: result.stats.topics_count },
                ]}
                columns={4}
              />

              {result.identity.content_analysis.tone && (
                <div className="pt-4 border-t">
                  <p className="text-sm text-gray-600">Communication Tone</p>
                  <p className="font-semibold capitalize">{result.identity.content_analysis.tone}</p>
                </div>
              )}

              {result.identity.content_analysis.personality_traits.length > 0 && (
                <div className="pt-4 border-t">
                  <p className="text-sm text-gray-600 mb-2">Personality Traits</p>
                  <TagsList tags={result.identity.content_analysis.personality_traits} variant="primary" />
                </div>
              )}

              {result.identity.content_analysis.topics.length > 0 && (
                <div className="pt-4 border-t">
                  <p className="text-sm text-gray-600 mb-2">Topics</p>
                  <TagsList tags={result.identity.content_analysis.topics} variant="gray" maxTags={10} />
                </div>
              )}

              <div className="pt-4 border-t">
                <Button onClick={handleViewIdentity} className="w-full">
                  View Full Identity
                </Button>
              </div>
            </div>
          </Card>
        )}
      </div>
    </PageLayout>
  );
};

export default BuildIdentityPage;

