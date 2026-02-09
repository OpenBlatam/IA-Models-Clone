'use client';

import { Info, Code, Users, Heart, ExternalLink } from 'lucide-react';

export default function AboutPanel() {
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg p-6 border border-gray-200 shadow-sm">
        <div className="flex items-center gap-2 mb-6">
          <Info className="w-5 h-5 text-tesla-blue" />
          <h3 className="text-lg font-semibold text-tesla-black">Acerca de</h3>
        </div>

        {/* App Info */}
        <div className="mb-6">
          <h4 className="text-xl font-bold text-tesla-black mb-2">Robot Movement AI</h4>
          <p className="text-tesla-gray-dark mb-4">
            Plataforma enterprise completa para el control y monitoreo de robots con capacidades
            avanzadas de IA, visualización 3D, análisis predictivo y mucho más.
          </p>
          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 bg-gray-50 rounded-md border border-gray-200">
              <p className="text-sm text-tesla-gray-dark mb-1 font-medium">Versión</p>
              <p className="text-tesla-black font-mono font-semibold">1.0.0</p>
            </div>
            <div className="p-3 bg-gray-50 rounded-md border border-gray-200">
              <p className="text-sm text-tesla-gray-dark mb-1 font-medium">Build</p>
              <p className="text-tesla-black font-mono font-semibold">2024.01.15</p>
            </div>
          </div>
        </div>

        {/* Features */}
        <div className="mb-6">
          <h4 className="text-lg font-semibold text-tesla-black mb-4">Características Principales</h4>
          <ul className="space-y-3">
            <li className="flex items-center gap-3 text-tesla-gray-dark">
              <Code className="w-4 h-4 text-tesla-blue flex-shrink-0" />
              <span>Control en tiempo real del robot</span>
            </li>
            <li className="flex items-center gap-3 text-tesla-gray-dark">
              <Code className="w-4 h-4 text-tesla-blue flex-shrink-0" />
              <span>Visualización 3D interactiva</span>
            </li>
            <li className="flex items-center gap-3 text-tesla-gray-dark">
              <Code className="w-4 h-4 text-tesla-blue flex-shrink-0" />
              <span>Análisis predictivo con IA</span>
            </li>
            <li className="flex items-center gap-3 text-tesla-gray-dark">
              <Code className="w-4 h-4 text-tesla-blue flex-shrink-0" />
              <span>Colaboración en tiempo real</span>
            </li>
            <li className="flex items-center gap-3 text-tesla-gray-dark">
              <Code className="w-4 h-4 text-tesla-blue flex-shrink-0" />
              <span>Sistema de reportes avanzado</span>
            </li>
            <li className="flex items-center gap-3 text-tesla-gray-dark">
              <Code className="w-4 h-4 text-tesla-blue flex-shrink-0" />
              <span>PWA instalable</span>
            </li>
          </ul>
        </div>

        {/* Tech Stack */}
        <div className="mb-6">
          <h4 className="text-lg font-semibold text-tesla-black mb-4">Stack Tecnológico</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {['Next.js', 'React', 'TypeScript', 'Tailwind CSS', 'Three.js', 'Zustand', 'Recharts', 'Axios'].map((tech) => (
              <div
                key={tech}
                className="p-3 bg-gray-50 rounded-md border border-gray-200 text-center hover:border-tesla-blue transition-colors"
              >
                <p className="text-sm text-tesla-black font-medium">{tech}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Links */}
        <div className="space-y-3">
          <h4 className="text-lg font-semibold text-tesla-black">Enlaces</h4>
          <div className="space-y-2">
            <a
              href="#"
              className="flex items-center gap-3 p-4 bg-white rounded-md border border-gray-200 hover:border-tesla-blue hover:shadow-sm transition-all min-h-[44px]"
            >
              <ExternalLink className="w-4 h-4 text-tesla-blue flex-shrink-0" />
              <span className="text-tesla-black font-medium">Documentación</span>
            </a>
            <a
              href="#"
              className="flex items-center gap-3 p-4 bg-white rounded-md border border-gray-200 hover:border-tesla-blue hover:shadow-sm transition-all min-h-[44px]"
            >
              <Users className="w-4 h-4 text-tesla-blue flex-shrink-0" />
              <span className="text-tesla-black font-medium">Soporte</span>
            </a>
            <a
              href="#"
              className="flex items-center gap-3 p-4 bg-white rounded-md border border-gray-200 hover:border-tesla-blue hover:shadow-sm transition-all min-h-[44px]"
            >
              <Heart className="w-4 h-4 text-tesla-blue flex-shrink-0" />
              <span className="text-tesla-black font-medium">Contribuir</span>
            </a>
          </div>
        </div>

        {/* Copyright */}
        <div className="mt-6 pt-6 border-t border-gray-200 text-center">
          <p className="text-sm text-tesla-gray-dark">
            © 2024 Robot Movement AI. Todos los derechos reservados.
          </p>
        </div>
      </div>
    </div>
  );
}


