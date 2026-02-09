'use client';

import { useState } from 'react';
import { Shield, User, Lock, Unlock } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface Permission {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
}

interface Role {
  id: string;
  name: string;
  permissions: string[];
}

export default function PermissionsManager() {
  const [permissions] = useState<Permission[]>([
    { id: '1', name: 'Control Robot', description: 'Permite controlar el movimiento del robot', enabled: true },
    { id: '2', name: 'Ver Métricas', description: 'Acceso a métricas y estadísticas', enabled: true },
    { id: '3', name: 'Editar Configuración', description: 'Modificar configuraciones del sistema', enabled: false },
    { id: '4', name: 'Gestionar Usuarios', description: 'Crear y editar usuarios', enabled: false },
    { id: '5', name: 'Exportar Datos', description: 'Exportar datos del sistema', enabled: true },
    { id: '6', name: 'Acceso API', description: 'Usar la API del sistema', enabled: false },
  ]);

  const [roles] = useState<Role[]>([
    { id: '1', name: 'Administrador', permissions: ['1', '2', '3', '4', '5', '6'] },
    { id: '2', name: 'Operador', permissions: ['1', '2', '5'] },
    { id: '3', name: 'Visualizador', permissions: ['2'] },
  ]);

  const [selectedRole, setSelectedRole] = useState<string>('1');

  const currentRole = roles.find((r) => r.id === selectedRole);

  const togglePermission = (permissionId: string) => {
    toast.info(`Permiso ${permissions.find((p) => p.id === permissionId)?.name} actualizado`);
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-6">
          <Shield className="w-5 h-5 text-primary-400" />
          <h3 className="text-lg font-semibold text-white">Gestión de Permisos</h3>
        </div>

        {/* Roles */}
        <div className="mb-6">
          <h4 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
            <User className="w-4 h-4" />
            Roles
          </h4>
          <div className="grid grid-cols-3 gap-3">
            {roles.map((role) => (
              <button
                key={role.id}
                onClick={() => setSelectedRole(role.id)}
                className={`p-3 rounded-lg border transition-colors ${
                  selectedRole === role.id
                    ? 'bg-primary-600/20 border-primary-500'
                    : 'bg-gray-700/50 border-gray-600 hover:border-gray-500'
                }`}
              >
                <p className="font-semibold text-white">{role.name}</p>
                <p className="text-xs text-gray-400 mt-1">
                  {role.permissions.length} permisos
                </p>
              </button>
            ))}
          </div>
        </div>

        {/* Permissions */}
        <div>
          <h4 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
            <Lock className="w-4 h-4" />
            Permisos para: {currentRole?.name}
          </h4>
          <div className="space-y-2">
            {permissions.map((permission) => {
              const hasPermission = currentRole?.permissions.includes(permission.id);
              return (
                <div
                  key={permission.id}
                  className={`p-4 rounded-lg border ${
                    hasPermission
                      ? 'bg-green-500/10 border-green-500/50'
                      : 'bg-gray-700/50 border-gray-600'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        {hasPermission ? (
                          <Unlock className="w-4 h-4 text-green-400" />
                        ) : (
                          <Lock className="w-4 h-4 text-gray-400" />
                        )}
                        <h5 className="font-semibold text-white">{permission.name}</h5>
                      </div>
                      <p className="text-sm text-gray-300">{permission.description}</p>
                    </div>
                    <button
                      onClick={() => togglePermission(permission.id)}
                      className={`px-3 py-1 rounded-lg text-sm transition-colors ${
                        hasPermission
                          ? 'bg-green-600 hover:bg-green-700 text-white'
                          : 'bg-gray-600 hover:bg-gray-700 text-gray-300'
                      }`}
                    >
                      {hasPermission ? 'Habilitado' : 'Deshabilitado'}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}


