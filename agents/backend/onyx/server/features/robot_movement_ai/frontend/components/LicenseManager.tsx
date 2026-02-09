'use client';

import { useState } from 'react';
import { Key, CheckCircle, XCircle, Clock, AlertTriangle } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface License {
  id: string;
  name: string;
  type: 'trial' | 'standard' | 'premium' | 'enterprise';
  status: 'active' | 'expired' | 'invalid';
  expiresAt?: Date;
  features: string[];
}

export default function LicenseManager() {
  const [license, setLicense] = useState<License | null>({
    id: '1',
    name: 'Robot Movement AI',
    type: 'trial',
    status: 'active',
    expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30 days
    features: ['Control básico', 'Chat', 'Visualización 3D'],
  });

  const [licenseKey, setLicenseKey] = useState('');

  const handleActivateLicense = () => {
    if (!licenseKey.trim()) {
      toast.error('Por favor ingresa una clave de licencia');
      return;
    }
    // Simulate license activation
    toast.success('Licencia activada exitosamente');
    setLicenseKey('');
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'expired':
        return <XCircle className="w-5 h-5 text-red-400" />;
      case 'invalid':
        return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
      default:
        return <Clock className="w-5 h-5 text-gray-400" />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'enterprise':
        return 'bg-purple-500/20 text-purple-400 border-purple-500/50';
      case 'premium':
        return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
      case 'standard':
        return 'bg-green-500/20 text-green-400 border-green-500/50';
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Key className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Gestión de Licencias</h3>
        </div>

        {license ? (
          <>
            {/* Current License */}
            <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    {getStatusIcon(license.status)}
                    <h4 className="font-semibold text-white">{license.name}</h4>
                    <span className={`px-2 py-1 rounded text-xs border ${getTypeColor(license.type)}`}>
                      {license.type}
                    </span>
                  </div>
                  <p className="text-sm text-gray-300">Estado: {license.status}</p>
                  {license.expiresAt && (
                    <p className="text-sm text-gray-300">
                      Expira: {license.expiresAt.toLocaleDateString('es-ES')}
                    </p>
                  )}
                </div>
              </div>

              <div>
                <h5 className="text-sm font-semibold text-white mb-2">Características:</h5>
                <ul className="space-y-1">
                  {license.features.map((feature, index) => (
                    <li key={index} className="text-sm text-gray-300 flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-green-400" />
                      {feature}
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Activate New License */}
            <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600">
              <h4 className="text-sm font-semibold text-white mb-3">Activar Nueva Licencia</h4>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={licenseKey}
                  onChange={(e) => setLicenseKey(e.target.value)}
                  placeholder="Ingresa la clave de licencia"
                  className="flex-1 px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <button
                  onClick={handleActivateLicense}
                  className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
                >
                  Activar
                </button>
              </div>
            </div>
          </>
        ) : (
          <div className="text-center py-12">
            <Key className="w-12 h-12 mx-auto mb-4 text-gray-400 opacity-50" />
            <p className="text-gray-400 mb-4">No hay licencia activa</p>
            <div className="p-4 bg-gray-700/50 rounded-lg border border-gray-600 max-w-md mx-auto">
              <h4 className="text-sm font-semibold text-white mb-3">Activar Licencia</h4>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={licenseKey}
                  onChange={(e) => setLicenseKey(e.target.value)}
                  placeholder="Ingresa la clave de licencia"
                  className="flex-1 px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <button
                  onClick={handleActivateLicense}
                  className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors"
                >
                  Activar
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


