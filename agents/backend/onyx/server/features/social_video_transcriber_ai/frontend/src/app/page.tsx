import Link from 'next/link';
import { Video, Sparkles, BarChart3, Zap } from 'lucide-react';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Video className="w-8 h-8 text-blue-600" />
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Social Video Transcriber AI
            </h1>
          </div>
          <nav className="flex gap-6">
            <Link href="/transcribe" className="text-gray-700 hover:text-blue-600 transition">
              Transcribir
            </Link>
            <Link href="/jobs" className="text-gray-700 hover:text-blue-600 transition">
              Trabajos
            </Link>
            <Link href="/batch" className="text-gray-700 hover:text-blue-600 transition">
              Batch
            </Link>
            <Link href="/analytics" className="text-gray-700 hover:text-blue-600 transition">
              Analytics
            </Link>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <main className="container mx-auto px-4 py-16">
        <div className="text-center max-w-4xl mx-auto mb-16">
          <h2 className="text-5xl font-bold mb-6 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Transcribe Videos con IA
          </h2>
          <p className="text-xl text-gray-600 mb-8">
            Transcribe videos de TikTok, Instagram y YouTube con análisis de IA avanzado,
            detección de frameworks y generación de variantes
          </p>
          <Link
            href="/transcribe"
            className="inline-block bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700 transition shadow-lg"
          >
            Comenzar Ahora
          </Link>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
          <FeatureCard
            icon={<Video className="w-8 h-8" />}
            title="Multi-Plataforma"
            description="Soporte para TikTok, Instagram y YouTube"
          />
          <FeatureCard
            icon={<Sparkles className="w-8 h-8" />}
            title="Análisis con IA"
            description="Detección de frameworks y estructura de contenido"
          />
          <FeatureCard
            icon={<BarChart3 className="w-8 h-8" />}
            title="Analytics"
            description="Métricas detalladas y estadísticas de uso"
          />
          <FeatureCard
            icon={<Zap className="w-8 h-8" />}
            title="Procesamiento Batch"
            description="Procesa múltiples videos simultáneamente"
          />
        </div>

        {/* Quick Actions */}
        <div className="bg-white rounded-xl shadow-lg p-8">
          <h3 className="text-2xl font-bold mb-6 text-center">Acciones Rápidas</h3>
          <div className="grid md:grid-cols-3 gap-4">
            <QuickActionCard
              title="Nueva Transcripción"
              description="Transcribe un video individual"
              href="/transcribe"
              color="blue"
            />
            <QuickActionCard
              title="Procesamiento Batch"
              description="Procesa múltiples videos"
              href="/batch"
              color="purple"
            />
            <QuickActionCard
              title="Ver Trabajos"
              description="Gestiona tus transcripciones"
              href="/jobs"
              color="green"
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t bg-white mt-16">
        <div className="container mx-auto px-4 py-8 text-center text-gray-600">
          <p>© 2024 Social Video Transcriber AI - Powered by Blatam Academy</p>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="bg-white rounded-lg p-6 shadow-md hover:shadow-lg transition">
      <div className="text-blue-600 mb-4">{icon}</div>
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p className="text-gray-600">{description}</p>
    </div>
  );
}

function QuickActionCard({
  title,
  description,
  href,
  color,
}: {
  title: string;
  description: string;
  href: string;
  color: 'blue' | 'purple' | 'green';
}) {
  const colorClasses = {
    blue: 'bg-blue-600 hover:bg-blue-700',
    purple: 'bg-purple-600 hover:bg-purple-700',
    green: 'bg-green-600 hover:bg-green-700',
  };

  return (
    <Link
      href={href}
      className={`${colorClasses[color]} text-white rounded-lg p-6 block transition shadow-md hover:shadow-lg`}
    >
      <h4 className="text-xl font-semibold mb-2">{title}</h4>
      <p className="text-sm opacity-90">{description}</p>
    </Link>
  );
}




