'use client';

import { useQuery } from 'react-query';
import { useParams, useRouter } from 'next/navigation';
import { apiClient } from '@/lib/api/client';
import type { IdentityProfile, GeneratedContent } from '@/types';
import PageLayout from '@/components/Layout/PageLayout';
import Card from '@/components/UI/Card';
import Button from '@/components/UI/Button';
import LoadingSpinner from '@/components/UI/LoadingSpinner';
import { formatDate } from '@/lib/utils';
import PlatformBadge from '@/components/UI/PlatformBadge';
import CopyButton from '@/components/UI/CopyButton';
import ExpandableSection from '@/components/UI/ExpandableSection';

const CONTENT_LIMIT = 20;

const IdentityDetailPage = (): JSX.Element => {
  const params = useParams();
  const router = useRouter();
  const identityId = params.id as string;

  const { data: identity, isLoading: identityLoading } = useQuery<IdentityProfile>(
    ['identity', identityId],
    () => apiClient.getIdentity(identityId),
    { enabled: !!identityId }
  );

  const { data: generatedContent, isLoading: contentLoading } = useQuery<GeneratedContent[]>(
    ['generated-content', identityId],
    () => apiClient.getGeneratedContent(identityId, CONTENT_LIMIT),
    { enabled: !!identityId }
  );

  const handleGenerateContent = (): void => {
    router.push('/generate-content');
  };

  const handleBackToIdentities = (): void => {
    router.push('/identities');
  };

  if (identityLoading) {
    return (
      <PageLayout>
        <LoadingSpinner />
      </PageLayout>
    );
  }

  if (!identity) {
    return (
      <PageLayout>
        <Card>
          <p className="text-center text-gray-600 py-8">Identity not found</p>
          <div className="text-center">
            <Button onClick={handleBackToIdentities}>
              Back to Identities
            </Button>
          </div>
        </Card>
      </PageLayout>
    );
  }

  return (
    <PageLayout
      breadcrumbs={[
        { label: 'Home', href: '/' },
        { label: 'Identities', href: '/identities' },
        { label: identity.username || identityId },
      ]}
    >
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold">{identity.username}</h1>
            {identity.display_name && (
              <p className="text-gray-600 mt-1">{identity.display_name}</p>
            )}
          </div>
          <Button onClick={handleGenerateContent} aria-label="Generate content for this identity">
            Generate Content
          </Button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-2 space-y-6">
            <Card title="Profile Information">
              <div className="space-y-4">
                {identity.bio && (
                  <div>
                    <p className="text-sm text-gray-600 mb-1">Bio</p>
                    <p>{identity.bio}</p>
                  </div>
                )}

                <div className="grid grid-cols-3 gap-4 pt-4 border-t">
                  <div>
                    <p className="text-sm text-gray-600">Videos</p>
                    <p className="text-2xl font-bold">{identity.total_videos}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Posts</p>
                    <p className="text-2xl font-bold">{identity.total_posts}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Comments</p>
                    <p className="text-2xl font-bold">{identity.total_comments}</p>
                  </div>
                </div>

                <div className="pt-4 border-t">
                  <p className="text-sm text-gray-600 mb-2">Platforms</p>
                  <div className="flex gap-2">
                    {identity.tiktok_profile && (
                      <PlatformBadge platform="tiktok" />
                    )}
                    {identity.instagram_profile && (
                      <PlatformBadge platform="instagram" />
                    )}
                    {identity.youtube_profile && (
                      <PlatformBadge platform="youtube" />
                    )}
                  </div>
                </div>
              </div>
            </Card>

            <Card title="Content Analysis">
              <div className="space-y-3">
                {identity.content_analysis.tone && (
                  <ExpandableSection title="Tone" defaultExpanded>
                    <p className="font-semibold capitalize">{identity.content_analysis.tone}</p>
                  </ExpandableSection>
                )}

                {identity.content_analysis.communication_style && (
                  <ExpandableSection title="Communication Style" defaultExpanded>
                    <p className="font-semibold">{identity.content_analysis.communication_style}</p>
                  </ExpandableSection>
                )}

                {identity.content_analysis.personality_traits.length > 0 && (
                  <ExpandableSection title="Personality Traits" defaultExpanded>
                    <div className="flex flex-wrap gap-2">
                      {identity.content_analysis.personality_traits.map((trait, idx) => (
                        <span
                          key={idx}
                          className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm"
                        >
                          {trait}
                        </span>
                      ))}
                    </div>
                  </ExpandableSection>
                )}

                {identity.content_analysis.topics.length > 0 && (
                  <ExpandableSection title="Topics">
                    <div className="flex flex-wrap gap-2">
                      {identity.content_analysis.topics.map((topic, idx) => (
                        <span
                          key={idx}
                          className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm"
                        >
                          {topic}
                        </span>
                      ))}
                    </div>
                  </ExpandableSection>
                )}

                {identity.content_analysis.values.length > 0 && (
                  <ExpandableSection title="Values">
                    <div className="flex flex-wrap gap-2">
                      {identity.content_analysis.values.map((value, idx) => (
                        <span
                          key={idx}
                          className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm"
                        >
                          {value}
                        </span>
                      ))}
                    </div>
                  </ExpandableSection>
                )}
              </div>
            </Card>
          </div>

          <div className="space-y-6">
            <Card title="Metadata">
              <div className="space-y-2 text-sm">
                <div>
                  <p className="text-gray-600">Created</p>
                  <p>{formatDate(identity.created_at)}</p>
                </div>
                <div>
                  <p className="text-gray-600">Updated</p>
                  <p>{formatDate(identity.updated_at)}</p>
                </div>
                <div>
                  <p className="text-gray-600 mb-2">Identity ID</p>
                  <div className="flex items-center gap-2">
                    <p className="font-mono text-xs break-all">{identity.profile_id}</p>
                    <CopyButton text={identity.profile_id} label="Copy ID" variant="secondary" className="text-xs" />
                  </div>
                </div>
                <div className="pt-4 border-t space-y-2">
                  <Button
                    variant="secondary"
                    onClick={() => router.push(`/identities/${identityId}/versions`)}
                    className="w-full text-sm"
                  >
                    View Versions
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={() => router.push(`/identities/${identityId}/analytics`)}
                    className="w-full text-sm"
                  >
                    View Analytics
                  </Button>
                  <div className="flex gap-2">
                    <Button
                      variant="secondary"
                      onClick={async () => {
                        const data = await apiClient.exportIdentityJSON(identityId);
                        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `identity-${identityId}.json`;
                        a.click();
                        URL.revokeObjectURL(url);
                      }}
                      className="flex-1 text-sm"
                    >
                      Export JSON
                    </Button>
                    <Button
                      variant="secondary"
                      onClick={async () => {
                        const csv = await apiClient.exportIdentityCSV(identityId);
                        const blob = new Blob([csv], { type: 'text/csv' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `identity-${identityId}.csv`;
                        a.click();
                        URL.revokeObjectURL(url);
                      }}
                      className="flex-1 text-sm"
                    >
                      Export CSV
                    </Button>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </div>

        <Card title="Generated Content">
          {contentLoading ? (
            <LoadingSpinner />
          ) : !generatedContent || generatedContent.length === 0 ? (
            <p className="text-center text-gray-600 py-8">No generated content yet</p>
          ) : (
            <div className="space-y-4">
              {generatedContent.map((content) => (
                <div
                  key={content.content_id}
                  className="p-4 border rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <PlatformBadge platform={content.platform} className="text-xs px-2 py-1" />
                      <span className="text-xs text-gray-600" aria-label="Content type">
                        {content.content_type}
                      </span>
                    </div>
                    <span className="text-xs text-gray-500">
                      {formatDate(content.generated_at)}
                    </span>
                  </div>
                  {content.title && (
                    <h4 className="font-semibold mb-2">{content.title}</h4>
                  )}
                  <p className="text-sm text-gray-700 line-clamp-3">{content.content}</p>
                  {content.hashtags.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-2">
                      {content.hashtags.slice(0, 5).map((tag, idx) => (
                        <span key={idx} className="text-xs text-primary-600">
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </PageLayout>
  );
};

export default IdentityDetailPage;

