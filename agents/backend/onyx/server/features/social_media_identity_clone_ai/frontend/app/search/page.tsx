'use client';

import { useState } from 'react';
import { useQuery } from 'react-query';
import { apiClient } from '@/lib/api/client';
import type { IdentityProfile, GeneratedContent } from '@/types';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import SearchInput from '@/components/UI/SearchInput';
import Tabs from '@/components/UI/Tabs';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import EmptyState from '@/components/UI/EmptyState';
import { formatDate } from '@/lib/utils';
import PlatformBadge from '@/components/UI/PlatformBadge';

const SearchPage = (): JSX.Element => {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState<'identities' | 'content'>('identities');

  const { data: identities, isLoading: identitiesLoading } = useQuery<IdentityProfile[]>(
    ['search-identities', searchQuery],
    () => apiClient.searchIdentities({ query: searchQuery }),
    { enabled: activeTab === 'identities' && searchQuery.length > 0 }
  );

  const { data: content, isLoading: contentLoading } = useQuery<GeneratedContent[]>(
    ['search-content', searchQuery],
    () => apiClient.searchContent(searchQuery),
    { enabled: activeTab === 'content' && searchQuery.length > 0 }
  );

  const handleSearch = (query: string): void => {
    setSearchQuery(query);
  };

  const tabs = [
    {
      id: 'identities',
      label: 'Identities',
      content: (
        <div className="space-y-4">
          {identitiesLoading ? (
            <LoadingSpinner />
          ) : !identities || identities.length === 0 ? (
            <EmptyState
              title="No identities found"
              description={searchQuery ? `No results for "${searchQuery}"` : 'Start typing to search'}
            />
          ) : (
            identities.map((identity) => (
              <Card key={identity.profile_id}>
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-xl font-semibold">{identity.username}</h3>
                    {identity.display_name && (
                      <p className="text-gray-600">{identity.display_name}</p>
                    )}
                  </div>
                  <div className="flex gap-2">
                    {identity.tiktok_profile && <PlatformBadge platform="tiktok" />}
                    {identity.instagram_profile && <PlatformBadge platform="instagram" />}
                    {identity.youtube_profile && <PlatformBadge platform="youtube" />}
                  </div>
                </div>
              </Card>
            ))
          )}
        </div>
      ),
    },
    {
      id: 'content',
      label: 'Content',
      content: (
        <div className="space-y-4">
          {contentLoading ? (
            <LoadingSpinner />
          ) : !content || content.length === 0 ? (
            <EmptyState
              title="No content found"
              description={searchQuery ? `No results for "${searchQuery}"` : 'Start typing to search'}
            />
          ) : (
            content.map((item) => (
              <Card key={item.content_id}>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <PlatformBadge platform={item.platform} />
                    <span className="text-sm text-gray-500">{formatDate(item.generated_at)}</span>
                  </div>
                  {item.title && <h4 className="font-semibold">{item.title}</h4>}
                  <p className="text-sm text-gray-700 line-clamp-3">{item.content}</p>
                </div>
              </Card>
            ))
          )}
        </div>
      ),
    },
  ];

  return (
    <PageLayout>
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Search</h1>
        <Card className="mb-6">
          <SearchInput
            onSearch={handleSearch}
            placeholder="Search identities or content..."
            className="w-full"
          />
        </Card>
        <Tabs tabs={tabs} defaultTab={activeTab} />
      </div>
    </PageLayout>
  );
};

export default SearchPage;



