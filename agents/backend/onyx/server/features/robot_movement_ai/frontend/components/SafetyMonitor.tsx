'use client';

import { useState, useEffect } from 'react';
import { useRobotStore } from '@/lib/store/robotStore';
import { Shield, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

interface SafetyCheck {
  name: string;
  status: 'safe' | 'warning' | 'danger';
  message: string;
  value?: number;
  threshold?: number;
}

export default function SafetyMonitor() {
  const { status } = useRobotStore();
  const [safetyChecks, setSafetyChecks] = useState<SafetyCheck[]>([]);

  useEffect(() => {
    const checks: SafetyCheck[] = [];

    // Velocity check
    if (status?.robot_status.velocity) {
      const velocity = Math.sqrt(
        status.robot_status.velocity[0] ** 2 +
        status.robot_status.velocity[1] ** 2 +
        status.robot_status.velocity[2] ** 2
      );
      checks.push({
        name: 'Velocidad',
        status: velocity > 1.0 ? 'danger' : velocity > 0.8 ? 'warning' : 'safe',
        message: velocity > 1.0 ? 'Velocidad excedida' : velocity > 0.8 ? 'Velocidad alta' : 'Velocidad segura',
        value: velocity,
        threshold: 1.0,
      });
    }

    // Position bounds check
    if (status?.robot_status.position) {
      const { x, y, z } = status.robot_status.position;
      const inBounds = Math.abs(x) < 2 && Math.abs(y) < 2 && Math.abs(z) < 2;
      checks.push({
        name: 'Límites de Posición',
        status: inBounds ? 'safe' : 'danger',
        message: inBounds ? 'Dentro de límites' : 'Fuera de límites seguros',
      });
    }

    // Connection check
    checks.push({
      name: 'Conexión',
      status: status?.robot_status.connected ? 'safe' : 'danger',
      message: status?.robot_status.connected ? 'Conectado' : 'Desconectado',
    });

    // Movement check
    checks.push({
      name: 'Estado de Movimiento',
      status: status?.robot_status.moving ? 'warning' : 'safe',
      message: status?.robot_status.moving ? 'En movimiento' : 'Detenido',
    });

    // Error check
    checks.push({
      name: 'Errores',
      status: status?.robot_status.error ? 'danger' : 'safe',
      message: status?.robot_status.error || 'Sin errores',
    });

    setSafetyChecks(checks);
  }, [status]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'safe':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
      case 'danger':
        return <XCircle className="w-5 h-5 text-red-400" />;
      default:
        return <Shield className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'safe':
        return 'border-green-500/50 bg-green-500/10';
      case 'warning':
        return 'border-yellow-500/50 bg-yellow-500/10';
      case 'danger':
        return 'border-red-500/50 bg-red-500/10';
      default:
        return 'border-gray-500/50 bg-gray-500/10';
    }
  };

  const safeCount = safetyChecks.filter((c) => c.status === 'safe').length;
  const warningCount = safetyChecks.filter((c) => c.status === 'warning').length;
  const dangerCount = safetyChecks.filter((c) => c.status === 'danger').length;

  return (
    <div className="space-y-6">
      {/* Summary */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-4">
          <Shield className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Monitor de Seguridad</h3>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div className="p-4 bg-green-500/10 border border-green-500/50 rounded-lg">
            <p className="text-sm text-gray-400 mb-1">Seguro</p>
            <p className="text-2xl font-bold text-green-400">{safeCount}</p>
          </div>
          <div className="p-4 bg-yellow-500/10 border border-yellow-500/50 rounded-lg">
            <p className="text-sm text-gray-400 mb-1">Advertencias</p>
            <p className="text-2xl font-bold text-yellow-400">{warningCount}</p>
          </div>
          <div className="p-4 bg-red-500/10 border border-red-500/50 rounded-lg">
            <p className="text-sm text-gray-400 mb-1">Peligro</p>
            <p className="text-2xl font-bold text-red-400">{dangerCount}</p>
          </div>
        </div>
      </div>

      {/* Safety Checks */}
      <div className="space-y-3">
        {safetyChecks.map((check, index) => (
          <div
            key={index}
            className={`p-4 rounded-lg border ${getStatusColor(check.status)}`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {getStatusIcon(check.status)}
                <div>
                  <h4 className="font-semibold text-white">{check.name}</h4>
                  <p className="text-sm text-gray-300">{check.message}</p>
                  {check.value !== undefined && check.threshold && (
                    <div className="mt-2">
                      <div className="flex justify-between text-xs text-gray-400 mb-1">
                        <span>Valor: {check.value.toFixed(3)}</span>
                        <span>Límite: {check.threshold}</span>
                      </div>
                      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all ${
                            check.status === 'danger'
                              ? 'bg-red-400'
                              : check.status === 'warning'
                              ? 'bg-yellow-400'
                              : 'bg-green-400'
                          }`}
                          style={{
                            width: `${Math.min((check.value / check.threshold) * 100, 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}


