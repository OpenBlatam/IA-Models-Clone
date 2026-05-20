'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiFileText, FiBriefcase, FiBarChart, FiTarget, FiBook, FiZap } from 'react-icons/fi';

export interface Template {
  id: string;
  name: string;
  description: string;
  icon: React.ComponentType<{ size?: number; className?: string }>;
  query: string;
  business_area: string;
  document_type: string;
}

const templates: Template[] = [
  {
    id: 'marketing-plan',
    name: 'Plan de Marketing',
    description: 'Estrategia completa de marketing digital',
    icon: FiTarget,
    query: 'Necesito un plan de marketing completo para lanzar un nuevo producto. Incluye estrategia digital, redes sociales, contenido y métricas de éxito.',
    business_area: 'marketing',
    document_type: 'plan',
  },
  {
    id: 'business-proposal',
    name: 'Propuesta Comercial',
    description: 'Propuesta profesional para clientes',
    icon: FiBriefcase,
    query: 'Crear una propuesta comercial profesional que incluya presentación de la empresa, servicios ofrecidos, casos de éxito y propuesta de valor.',
    business_area: 'ventas',
    document_type: 'propuesta',
  },
  {
    id: 'financial-report',
    name: 'Reporte Financiero',
    description: 'Análisis financiero detallado',
    icon: FiBarChart,
    query: 'Generar un reporte financiero que analice los ingresos, gastos, proyecciones y recomendaciones para el próximo trimestre.',
    business_area: 'finanzas',
    document_type: 'reporte',
  },
  {
    id: 'hr-policy',
    name: 'Política de RRHH',
    description: 'Documento de políticas de recursos humanos',
    icon: FiBook,
    query: 'Crear un documento de políticas de recursos humanos que cubra código de conducta, beneficios, procedimientos y valores de la empresa.',
    business_area: 'recursos-humanos',
    document_type: 'documento',
  },
  {
    id: 'tech-strategy',
    name: 'Estrategia Tecnológica',
    description: 'Plan estratégico de tecnología',
    icon: FiZap,
    query: 'Desarrollar una estrategia tecnológica que incluya infraestructura, herramientas, seguridad, innovación y roadmap a 3 años.',
    business_area: 'tecnologia',
    document_type: 'estrategia',
  },
  {
    id: 'operations-manual',
    name: 'Manual de Operaciones',
    description: 'Guía completa de operaciones',
    icon: FiFileText,
    query: 'Crear un manual de operaciones detallado con procesos, procedimientos, responsabilidades y flujos de trabajo del departamento.',
    business_area: 'operaciones',
    document_type: 'documento',
  },
];

interface TemplatesPanelProps {
  onSelectTemplate: (template: Template) => void;
}

export default function TemplatesPanel({ onSelectTemplate }: TemplatesPanelProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="btn btn-secondary mb-4"
        data-help="templates"
      >
        <FiFileText size={18} />
        Usar Plantilla
      </button>

      <AnimatePresence>
        {isOpen && (
          <>
            <div
              className="fixed inset-0 bg-black bg-opacity-50 z-40"
              onClick={() => setIsOpen(false)}
            />
            <motion.div
              initial={{ opacity: 0, x: 300 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 300 }}
              className="fixed right-0 top-0 h-full w-96 bg-white dark:bg-gray-900 shadow-xl z-50 overflow-y-auto"
            >
              <div className="p-6 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                    Plantillas
                  </h3>
                  <button
                    onClick={() => setIsOpen(false)}
                    className="btn-icon"
                  >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M18 6L6 18" />
                      <path d="M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Selecciona una plantilla para comenzar rápidamente
                </p>
              </div>

              <div className="p-4 space-y-3">
                {templates.map((template) => (
                  <motion.button
                    key={template.id}
                    onClick={() => {
                      onSelectTemplate(template);
                      setIsOpen(false);
                    }}
                    className="w-full text-left p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:border-primary-500 hover:bg-primary-50 dark:hover:bg-primary-900/20 transition-colors"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div className="flex items-start gap-3">
                      <div className="p-2 bg-primary-100 dark:bg-primary-900 rounded-lg">
                        <template.icon className="text-primary-600 dark:text-primary-400" size={20} />
                      </div>
                      <div className="flex-1">
                        <h4 className="font-semibold text-gray-900 dark:text-white mb-1">
                          {template.name}
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {template.description}
                        </p>
                      </div>
                    </div>
                  </motion.button>
                ))}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
