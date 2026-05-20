'use client';

export const dynamic = 'force-dynamic';

import { useEffect, useState } from 'react';
import { Layout } from '@/components/layout/Layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Loading } from '@/components/ui/Loading';
import { Alert } from '@/components/ui/Alert';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';
import { Badge } from '@/components/ui/Badge';
import { analyticsApi } from '@/lib/api';
import { Analytics } from '@/types';
import { getPlatformIcon } from '@/lib/utils';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp } from 'lucide-react';

const PLATFORMS = ['facebook', 'instagram', 'twitter', 'linkedin', 'tiktok', 'youtube'];

const COLORS = ['#0ea5e9', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#ef4444'];

export default function AnalyticsPage() {
  const [selectedPlatform, setSelectedPlatform] = useState<string>('all');
  const [platformAnalytics, setPlatformAnalytics] = useState<Record<string, Analytics>>({});
  const [trends, setTrends] = useState<any[]>([]);
  const [bestPerforming, setBestPerforming] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [days, setDays] = useState(7);

  useEffect(() => {
    fetchAnalytics();
  }, [selectedPlatform, days]);

  const fetchAnalytics = async () => {
    setLoading(true);
    setError(null);

    try {
      if (selectedPlatform === 'all') {
        const allAnalytics: Record<string, Analytics> = {};
        for (const platform of PLATFORMS) {
          try {
            const data = await analyticsApi.getPlatformAnalytics(platform, days);
            allAnalytics[platform] = data;
          } catch (err) {
            console.error(`Error fetching analytics for ${platform}:`, err);
          }
        }
        setPlatformAnalytics(allAnalytics);
      } else {
        const [analytics, trendsData, bestPerformingData] = await Promise.all([
          analyticsApi.getPlatformAnalytics(selectedPlatform, days),
          analyticsApi.getTrends(selectedPlatform, days),
          analyticsApi.getBestPerforming(selectedPlatform, 10),
        ]);
        setPlatformAnalytics({ [selectedPlatform]: analytics });
        setTrends(trendsData.trends || []);
        setBestPerforming(bestPerformingData);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Error al cargar analytics';
      setError(errorMessage);
      console.error('Error fetching analytics:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <Loading size="lg" text="Cargando analytics..." />
        </div>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout>
        <div className="space-y-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Analytics</h1>
            <p className="mt-2 text-gray-600">Métricas y análisis de rendimiento</p>
          </div>
          <Alert variant="error" title="Error al cargar datos">
            {error}
          </Alert>
        </div>
      </Layout>
    );
  }

  const trendsData = trends.map((item) => ({
    date: new Date(item.date).toLocaleDateString('es-ES', { month: 'short', day: 'numeric' }),
    engagement: item.engagement || 0,
  }));

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Analytics</h1>
            <p className="mt-2 text-gray-600">Métricas y análisis de rendimiento</p>
          </div>
          <div className="flex items-center gap-4">
            <select
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500"
              aria-label="Período de tiempo"
            >
              <option value={7}>Últimos 7 días</option>
              <option value={30}>Últimos 30 días</option>
              <option value={90}>Últimos 90 días</option>
            </select>
            <select
              value={selectedPlatform}
              onChange={(e) => setSelectedPlatform(e.target.value)}
              className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500"
              aria-label="Plataforma"
            >
              <option value="all">Todas las plataformas</option>
              {PLATFORMS.map((platform) => (
                <option key={platform} value={platform}>
                  {getPlatformIcon(platform)} {platform}
                </option>
              ))}
            </select>
          </div>
        </div>

        <Tabs defaultValue="overview" className="w-full">
          <TabsList>
            <TabsTrigger value="overview">Resumen</TabsTrigger>
            <TabsTrigger value="trends">Tendencias</TabsTrigger>
            <TabsTrigger value="performance">Rendimiento</TabsTrigger>
          </TabsList>

          <TabsContent value="overview">
            {selectedPlatform === 'all' ? (
              <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
                {PLATFORMS.map((platform) => {
                  const analytics = platformAnalytics[platform];
                  if (!analytics) return null;

                  return (
                    <Card key={platform}>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          {getPlatformIcon(platform)} {platform}
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          <div>
                            <p className="text-sm text-gray-600">Total Posts</p>
                            <p className="text-2xl font-bold">{analytics.total_posts || 0}</p>
                          </div>
                          <div>
                            <p className="text-sm text-gray-600">Total Engagement</p>
                            <p className="text-2xl font-bold">{analytics.total_engagement || 0}</p>
                          </div>
                          <div>
                            <p className="text-sm text-gray-600">Engagement Rate</p>
                            <div className="flex items-center gap-2">
                              <TrendingUp className="h-4 w-4 text-green-600" />
                              <p className="text-2xl font-bold">
                                {(analytics.average_engagement_rate || 0).toFixed(2)}%
                              </p>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            ) : (
              <>
                <div className="grid grid-cols-1 gap-6 sm:grid-cols-3 mb-6">
                  {platformAnalytics[selectedPlatform] && (
                    <>
                      <Card>
                        <CardHeader>
                          <CardTitle>Total Posts</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="text-3xl font-bold">
                            {platformAnalytics[selectedPlatform].total_posts || 0}
                          </p>
                        </CardContent>
                      </Card>
                      <Card>
                        <CardHeader>
                          <CardTitle>Total Engagement</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="text-3xl font-bold">
                            {platformAnalytics[selectedPlatform].total_engagement || 0}
                          </p>
                        </CardContent>
                      </Card>
                      <Card>
                        <CardHeader>
                          <CardTitle>Engagement Rate</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="flex items-center gap-2">
                            <TrendingUp className="h-5 w-5 text-green-600" />
                            <p className="text-3xl font-bold">
                              {(platformAnalytics[selectedPlatform].average_engagement_rate || 0).toFixed(2)}%
                            </p>
                          </div>
                        </CardContent>
                      </Card>
                    </>
                  )}
                </div>

                {Object.keys(platformAnalytics).length > 0 && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Distribución por Plataforma</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <PieChart>
                          <Pie
                            data={Object.entries(platformAnalytics).map(([platform, data]) => ({
                              platform,
                              engagement: data.total_engagement || 0,
                            }))}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={({ platform, engagement }: { platform: string; engagement: number }) => `${platform}: ${engagement}`}
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="engagement"
                          >
                            {Object.keys(platformAnalytics).map((_, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                )}
              </>
            )}
          </TabsContent>

          <TabsContent value="trends">
            {selectedPlatform !== 'all' ? (
              <>
                <div className="grid grid-cols-1 gap-6 sm:grid-cols-3 mb-6">
                  {platformAnalytics[selectedPlatform] && (
                    <>
                      <Card>
                        <CardHeader>
                          <CardTitle>Total Posts</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="text-3xl font-bold">
                            {platformAnalytics[selectedPlatform].total_posts || 0}
                          </p>
                        </CardContent>
                      </Card>
                      <Card>
                        <CardHeader>
                          <CardTitle>Total Engagement</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="text-3xl font-bold">
                            {platformAnalytics[selectedPlatform].total_engagement || 0}
                          </p>
                        </CardContent>
                      </Card>
                      <Card>
                        <CardHeader>
                          <CardTitle>Engagement Rate</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="flex items-center gap-2">
                            <TrendingUp className="h-5 w-5 text-green-600" />
                            <p className="text-3xl font-bold">
                              {(platformAnalytics[selectedPlatform].average_engagement_rate || 0).toFixed(2)}%
                            </p>
                          </div>
                        </CardContent>
                      </Card>
                    </>
                  )}
                </div>

                {trendsData.length > 0 ? (
                  <Card>
                    <CardHeader>
                      <CardTitle>Tendencias de Engagement</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={400}>
                        <LineChart data={trendsData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" />
                          <YAxis />
                          <Tooltip />
                          <Line
                            type="monotone"
                            dataKey="engagement"
                            stroke="#0ea5e9"
                            strokeWidth={2}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                ) : (
                  <Card>
                    <CardContent className="py-12 text-center">
                      <p className="text-gray-500">No hay datos de tendencias disponibles</p>
                    </CardContent>
                  </Card>
                )}
              </>
            ) : (
              <Card>
                <CardContent className="py-12 text-center">
                  <p className="text-gray-500">Selecciona una plataforma para ver tendencias</p>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="performance">
            {bestPerforming.length > 0 ? (
              <Card>
                <CardHeader>
                  <CardTitle>Mejores Posts</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {bestPerforming.map((post, index) => (
                      <div
                        key={post.post_id || index}
                        className="flex items-center justify-between rounded-lg border border-gray-200 p-4 hover:bg-gray-50 transition-colors"
                      >
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <Badge variant="success" size="sm">
                              #{index + 1}
                            </Badge>
                            <p className="font-medium text-gray-900 line-clamp-2">
                              {post.content || 'Sin contenido'}
                            </p>
                          </div>
                          <div className="flex items-center gap-4 text-sm text-gray-500">
                            <span>Engagement: {post.engagement || 0}</span>
                            <span>Rate: {(post.engagement_rate || 0).toFixed(2)}%</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Card>
                <CardContent className="py-12 text-center">
                  <p className="text-gray-500">No hay datos de rendimiento disponibles</p>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </Layout>
  );
}

