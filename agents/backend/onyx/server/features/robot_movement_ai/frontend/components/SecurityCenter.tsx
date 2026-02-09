'use client';

import { useState } from 'react';
import { Shield, Lock, AlertTriangle, CheckCircle } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface SecurityCheck {
  id: string;
  name: string;
  status: 'secure' | 'warning' | 'critical';
  description: string;
  lastCheck: Date;
}

export default function SecurityCenter() {
  const [checks, setChecks] = useState<SecurityCheck[]>([
    {
      id: '1',
      name: 'Autenticación',
      status: 'secure',
      description: 'Sistema de autenticación activo y funcionando',
      lastCheck: new Date(),
    },
    {
      id: '2',
      name: 'Conexiones SSL',
      status: 'secure',
      description: 'Todas las conexiones están encriptadas',
      lastCheck: new Date(),
    },
    {
      id: '3',
      name: 'Firewall',
      status: 'warning',
      description: 'Algunas reglas del firewall necesitan revisión',
      lastCheck: new Date(Date.now() - 3600000),
    },
    {
      id: '4',
      name: 'Actualizaciones',
      status: 'critical',
      description: 'Hay actualizaciones de seguridad pendientes',
      lastCheck: new Date(Date.now() - 7200000),
    },
  ]);

  const handleRunCheck = (id: string) => {
    setChecks((prev) =>
      prev.map((c) =>
        c.id === id ? { ...c, lastCheck: new Date() } : c
      )
    );
    toast.success('Verificación de seguridad ejecutada');
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'secure':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
      case 'critical':
        return <AlertTriangle className="w-5 h-5 text-red-400" />;
      default:
        return <Shield className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'secure':
        return 'bg-green-500/10 border-green-500/50';
      case 'warning':
        return 'bg-yellow-500/10 border-yellow-500/50';
      case 'critical':
        return 'bg-red-500/10 border-red-500/50';
      default:
        return 'bg-gray-700/50 border-gray-600';
    }
  };

  const secureCount = checks.filter((c) => c.status === 'secure').length;
  const totalCount = checks.length;
  const securityScore = (secureCount / totalCount) * 100;

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Shield className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Centro de Seguridad</h3>
        </div>

        {/* Security Score */}
        <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-300">Puntuación de Seguridad</span>
            <span className="text-2xl font-bold text-white">{securityScore.toFixed(0)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-3">
            <div
              className={`h-3 rounded-full transition-all ${
                securityScore >= 80
                  ? 'bg-green-500'
                  : securityScore >= 50
                  ? 'bg-yellow-500'
                  : 'bg-red-500'
              }`}
              style={{ width: `${securityScore}%` }}
            />
          </div>
          <p className="text-xs text-gray-400 mt-2">
            {secureCount} de {totalCount} verificaciones seguras
          </p>
        </div>

        {/* Security Checks */}
        <div className="space-y-3">
          {checks.map((check) => (
            <div
              key={check.id}
              className={`p-4 rounded-lg border ${getStatusColor(check.status)}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3 flex-1">
                  {getStatusIcon(check.status)}
                  <div className="flex-1">
                    <h4 className="font-semibold text-white mb-1">{check.name}</h4>
                    <p className="text-sm text-gray-300">{check.description}</p>
                    <p className="text-xs text-gray-400 mt-1">
                      Última verificación: {check.lastCheck.toLocaleString('es-ES')}
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => handleRunCheck(check.id)}
                  className="px-3 py-1 bg-primary-600 hover:bg-primary-700 text-white text-sm rounded-lg transition-colors"
                >
                  Verificar
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


