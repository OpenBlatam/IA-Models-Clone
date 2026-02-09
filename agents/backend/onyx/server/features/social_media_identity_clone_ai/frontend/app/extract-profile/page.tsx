'use client';

import { useState } from 'react';
import { Platform, type ExtractProfileRequest, type ExtractProfileResponse } from '@/types';
import { useExtractProfileMutation } from '@/lib/modules/api';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import Input from '@/components/UI/Input';
import Select from '@/components/UI/Select';
import PlatformBadge from '@/components/UI/PlatformBadge';
import Checkbox from '@/components/UI/Checkbox';
import MutationError from '@/components/Forms/MutationError';
import MutationLoading from '@/components/Forms/MutationLoading';
import StatsGrid from '@/components/Display/StatsGrid';
import InfoRow from '@/components/Display/InfoRow';
import { getPlatformOptions } from '@/lib/modules/platform';

const ExtractProfilePage = (): JSX.Element => {
  const [platform, setPlatform] = useState<Platform>(Platform.INSTAGRAM);
  const [username, setUsername] = useState('');
  const [useCache, setUseCache] = useState(true);
  const [result, setResult] = useState<ExtractProfileResponse | null>(null);

  const extractMutation = useExtractProfileMutation();

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>): void => {
    e.preventDefault();
    
    const trimmedUsername = username.trim();
    if (!trimmedUsername) {
      return;
    }

    extractMutation.mutate(
      {
        platform,
        username: trimmedUsername,
        use_cache: useCache,
      },
      {
        onSuccess: (data) => {
          setResult(data);
        },
      }
    );
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLFormElement>): void => {
    if (e.key !== 'Enter' || !e.ctrlKey) {
      return;
    }
    
    e.preventDefault();
    const form = e.currentTarget;
    const formEvent = new Event('submit', { bubbles: true, cancelable: true }) as unknown as React.FormEvent<HTMLFormElement>;
    Object.defineProperty(formEvent, 'target', { value: form, enumerable: true });
    Object.defineProperty(formEvent, 'currentTarget', { value: form, enumerable: true });
    handleSubmit(formEvent);
  };

  const handlePlatformChange = (e: React.ChangeEvent<HTMLSelectElement>): void => {
    setPlatform(e.target.value as Platform);
  };

  const handleUsernameChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setUsername(e.target.value);
  };

  const handleCacheChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    setUseCache(e.target.checked);
  };

  return (
    <PageLayout>
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Extract Social Media Profile</h1>

        <Card title="Extract Profile">
          <form onSubmit={handleSubmit} onKeyDown={handleKeyDown}>
            <div className="space-y-4">
              <Select
                label="Platform"
                value={platform}
                onChange={handlePlatformChange}
                options={getPlatformOptions()}
              />

              <Input
                label="Username / Channel ID"
                type="text"
                value={username}
                onChange={handleUsernameChange}
                placeholder="Enter username or channel ID"
                required
              />

              <Checkbox
                id="useCache"
                label="Use cache if available"
                checked={useCache}
                onChange={setUseCache}
              />

              <Button type="submit" isLoading={extractMutation.isLoading} className="w-full">
                Extract Profile
              </Button>
            </div>
          </form>
        </Card>

        <MutationLoading isLoading={extractMutation.isLoading} />
        <MutationError error={extractMutation.error} />

        {result && (
          <Card className="mt-6" title="Extraction Results">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <PlatformBadge platform={result.platform} />
                <div className="text-sm text-gray-600">@{result.username}</div>
              </div>

              {result.profile.display_name && (
                <InfoRow label="Display Name" value={result.profile.display_name} />
              )}

              {result.profile.bio && <InfoRow label="Bio" value={result.profile.bio} />}

              <StatsGrid
                stats={[
                  { label: 'Videos', value: result.stats.videos },
                  { label: 'Posts', value: result.stats.posts },
                  { label: 'Comments', value: result.stats.comments },
                ]}
                columns={3}
              />

              {result.profile.followers_count !== undefined && (
                <div className="pt-4 border-t">
                  <p className="text-sm text-gray-600">Followers</p>
                  <p className="text-xl font-semibold">{result.profile.followers_count.toLocaleString()}</p>
                </div>
              )}
            </div>
          </Card>
        )}
      </div>
    </PageLayout>
  );
};

export default ExtractProfilePage;

