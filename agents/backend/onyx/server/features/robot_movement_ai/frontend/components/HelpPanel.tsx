'use client';

import {
  HelpCircle,
  Book,
  Video,
  Code,
  MessageSquare,
} from 'lucide-react';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/Accordion';

interface HelpSection {
  id: string;
  title: string;
  icon: React.ReactNode;
  content: string;
  items?: string[];
}

const helpSections: HelpSection[] = [
  {
    id: 'getting-started',
    title: 'Empezar',
    icon: <Book className="w-5 h-5" />,
    content: 'Guía rápida para comenzar a usar el sistema',
    items: [
      'Conecta el robot al backend',
      'Verifica el estado en la pestaña Estado',
      'Usa Control para mover el robot',
      'Prueba comandos en Chat',
    ],
  },
  {
    id: 'commands',
    title: 'Comandos Disponibles',
    icon: <Code className="w-5 h-5" />,
    content: 'Lista de comandos que puedes usar',
    items: [
      'move to (x, y, z) - Mover a posición absoluta',
      'move relative (x, y, z) - Mover relativamente',
      'go home - Ir a posición home',
      'stop - Detener movimiento',
      'status - Obtener estado del robot',
    ],
  },
  {
    id: 'shortcuts',
    title: 'Atajos de Teclado',
    icon: <MessageSquare className="w-5 h-5" />,
    content: 'Atajos rápidos para operaciones comunes',
    items: [
      'Ctrl + H - Ir a posición home',
      'Ctrl + S - Detener robot',
      'Ctrl + R - Iniciar/Detener grabación',
    ],
  },
  {
    id: 'features',
    title: 'Características',
    icon: <Video className="w-5 h-5" />,
    content: 'Funcionalidades principales del sistema',
    items: [
      'Visualización 3D del robot',
      'Grabación y reproducción de movimientos',
      'Optimización de trayectorias',
      'Métricas y análisis en tiempo real',
      'Sistema de alertas',
      'Logs del sistema',
    ],
  },
];

export default function HelpPanel() {

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg p-6 border border-gray-200 shadow-sm">
        <div className="flex items-center gap-2 mb-4">
          <HelpCircle className="w-6 h-6 text-tesla-blue" />
          <h2 className="text-2xl font-bold text-tesla-black">Centro de Ayuda</h2>
        </div>
        <p className="text-tesla-gray-dark">
          Encuentra información sobre cómo usar el sistema, comandos disponibles y más.
        </p>
      </div>

      <Accordion type="single" collapsible defaultValue="getting-started" className="space-y-4">
        {helpSections.map((section) => (
          <AccordionItem key={section.id} value={section.id} className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden transition-all hover:shadow-tesla-md">
            <AccordionTrigger className="px-5 hover:bg-gray-50">
              <div className="flex items-center gap-3">
                <div className="text-tesla-blue">{section.icon}</div>
                <div className="text-left">
                  <h3 className="font-semibold text-tesla-black">{section.title}</h3>
                  <p className="text-sm text-tesla-gray-dark">{section.content}</p>
                </div>
              </div>
            </AccordionTrigger>
            {section.items && (
              <AccordionContent className="px-5">
                <ul className="space-y-3">
                  {section.items.map((item, index) => (
                    <li key={index} className="flex items-start gap-3 text-tesla-gray-dark">
                      <span className="text-tesla-blue mt-1 font-bold">•</span>
                      <span className="flex-1">{item}</span>
                    </li>
                  ))}
                </ul>
              </AccordionContent>
            )}
          </AccordionItem>
        ))}
      </Accordion>

      {/* Quick Tips */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="font-semibold text-tesla-blue mb-4 flex items-center gap-2">
          <span className="text-xl">💡</span>
          Consejos Rápidos
        </h3>
        <ul className="space-y-2 text-sm text-tesla-gray-dark">
          <li className="flex items-start gap-2">
            <span className="text-tesla-blue mt-1">•</span>
            <span>Usa la visualización 3D para ver el robot en tiempo real</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-tesla-blue mt-1">•</span>
            <span>Graba movimientos para reproducirlos después</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-tesla-blue mt-1">•</span>
            <span>Compara algoritmos de optimización para elegir el mejor</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-tesla-blue mt-1">•</span>
            <span>Revisa los logs para diagnosticar problemas</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-tesla-blue mt-1">•</span>
            <span>Configura alertas para estar informado</span>
          </li>
        </ul>
      </div>
    </div>
  );
}

