'use client';

export const dynamic = 'force-dynamic';

import { useEffect, useState } from 'react';
import { Layout } from '@/components/layout/Layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Modal } from '@/components/ui/Modal';
import { platformsApi } from '@/lib/api';
import { Platform, PlatformConnect } from '@/types';
import { getPlatformIcon, getPlatformColor } from '@/lib/utils';
import { Plus, X, Check } from 'lucide-react';
import { useForm } from 'react-hook-form';

const PLATFORMS = [
  { id: 'facebook', name: 'Facebook', icon: '📘' },
  { id: 'instagram', name: 'Instagram', icon: '📷' },
  { id: 'twitter', name: 'Twitter/X', icon: '🐦' },
  { id: 'linkedin', name: 'LinkedIn', icon: '💼' },
  { id: 'tiktok', name: 'TikTok', icon: '🎵' },
  { id: 'youtube', name: 'YouTube', icon: '📺' },
];

export default function PlatformsPage() {
  const [platforms, setPlatforms] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedPlatform, setSelectedPlatform] = useState<string>('');
  const { register, handleSubmit, reset, formState: { errors } } = useForm<PlatformConnect>();

  useEffect(() => {
    fetchPlatforms();
  }, []);

  const fetchPlatforms = async () => {
    try {
      const data = await platformsApi.getAll();
      setPlatforms(data);
    } catch (error) {
      console.error('Error fetching platforms:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleConnect = async (data: PlatformConnect) => {
    try {
      await platformsApi.connect(data);
      setIsModalOpen(false);
      reset();
      fetchPlatforms();
    } catch (error) {
      console.error('Error connecting platform:', error);
      alert('Error al conectar la plataforma. Verifica las credenciales.');
    }
  };

  const handleDisconnect = async (platform: string) => {
    if (!confirm(`¿Estás seguro de desconectar ${platform}?`)) return;
    try {
      await platformsApi.disconnect(platform);
      fetchPlatforms();
    } catch (error) {
      console.error('Error disconnecting platform:', error);
    }
  };

  const openConnectModal = (platform: string) => {
    setSelectedPlatform(platform);
    reset({
      platform,
      credentials: {},
    });
    setIsModalOpen(true);
  };

  if (loading) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">Cargando...</div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Plataformas</h1>
            <p className="mt-2 text-gray-600">Gestiona tus conexiones a redes sociales</p>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {PLATFORMS.map((platform) => {
            const isConnected = platforms.includes(platform.id);

            return (
              <Card key={platform.id}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-3xl">{platform.icon}</span>
                      <CardTitle>{platform.name}</CardTitle>
                    </div>
                    {isConnected && (
                      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-green-100">
                        <Check className="h-4 w-4 text-green-600" />
                      </span>
                    )}
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    {isConnected ? (
                      <Button
                        variant="danger"
                        size="sm"
                        className="w-full"
                        onClick={() => handleDisconnect(platform.id)}
                      >
                        Desconectar
                      </Button>
                    ) : (
                      <Button
                        variant="primary"
                        size="sm"
                        className="w-full"
                        onClick={() => openConnectModal(platform.id)}
                      >
                        Conectar
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>

      <Modal
        isOpen={isModalOpen}
        onClose={() => {
          setIsModalOpen(false);
          reset();
        }}
        title={`Conectar ${PLATFORMS.find((p) => p.id === selectedPlatform)?.name}`}
        size="md"
      >
        <form onSubmit={handleSubmit(handleConnect)} className="space-y-4">
          <input type="hidden" {...register('platform')} value={selectedPlatform} />

          <div>
            <label htmlFor="access_token" className="block text-sm font-medium text-gray-700 mb-1">
              Access Token / API Key
            </label>
            <input
              id="access_token"
              type="text"
              {...register('credentials.access_token', { required: 'El token es requerido' })}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500"
              placeholder="Ingresa tu access token o API key"
            />
            {errors.credentials?.access_token && (
              <p className="mt-1 text-sm text-red-600">
                {errors.credentials.access_token.message}
              </p>
            )}
          </div>

          <div>
            <label htmlFor="secret" className="block text-sm font-medium text-gray-700 mb-1">
              Secret / API Secret (opcional)
            </label>
            <input
              id="secret"
              type="password"
              {...register('credentials.secret')}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500"
              placeholder="Ingresa tu secret o API secret"
            />
          </div>

          <div className="rounded-lg bg-yellow-50 p-4">
            <p className="text-sm text-yellow-800">
              <strong>Nota:</strong> Las credenciales se almacenan de forma segura. Asegúrate de
              tener los permisos necesarios para publicar en la plataforma.
            </p>
          </div>

          <div className="flex justify-end gap-2 pt-4">
            <Button
              type="button"
              variant="secondary"
              onClick={() => {
                setIsModalOpen(false);
                reset();
              }}
            >
              Cancelar
            </Button>
            <Button type="submit" variant="primary">
              Conectar
            </Button>
          </div>
        </form>
      </Modal>
    </Layout>
  );
}

