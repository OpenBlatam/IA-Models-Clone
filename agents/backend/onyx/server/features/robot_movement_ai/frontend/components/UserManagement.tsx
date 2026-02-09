'use client';

import { useState } from 'react';
import { Users, Plus, Edit, Trash2, Shield } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

interface User {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'operator' | 'viewer';
  status: 'active' | 'inactive';
  lastLogin?: Date;
}

export default function UserManagement() {
  const [users, setUsers] = useState<User[]>([
    {
      id: '1',
      name: 'Admin User',
      email: 'admin@example.com',
      role: 'admin',
      status: 'active',
      lastLogin: new Date(),
    },
    {
      id: '2',
      name: 'Operator User',
      email: 'operator@example.com',
      role: 'operator',
      status: 'active',
      lastLogin: new Date(Date.now() - 3600000),
    },
    {
      id: '3',
      name: 'Viewer User',
      email: 'viewer@example.com',
      role: 'viewer',
      status: 'inactive',
    },
  ]);
  const [showAdd, setShowAdd] = useState(false);
  const [newName, setNewName] = useState('');
  const [newEmail, setNewEmail] = useState('');
  const [newRole, setNewRole] = useState<'admin' | 'operator' | 'viewer'>('viewer');

  const handleAdd = () => {
    if (!newName.trim() || !newEmail.trim()) {
      toast.error('Nombre y email son requeridos');
      return;
    }

    const newUser: User = {
      id: Date.now().toString(),
      name: newName,
      email: newEmail,
      role: newRole,
      status: 'active',
    };

    setUsers([...users, newUser]);
    setNewName('');
    setNewEmail('');
    setShowAdd(false);
    toast.success('Usuario creado');
  };

  const handleDelete = (id: string) => {
    setUsers(users.filter((u) => u.id !== id));
    toast.success('Usuario eliminado');
  };

  const handleToggleStatus = (id: string) => {
    setUsers((prev) =>
      prev.map((u) =>
        u.id === id ? { ...u, status: u.status === 'active' ? 'inactive' : 'active' } : u
      )
    );
    toast.success('Estado actualizado');
  };

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'admin':
        return 'bg-red-500/20 text-red-400 border-red-500/50';
      case 'operator':
        return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Users className="w-5 h-5 text-primary-400" />
            <h3 className="text-lg font-semibold text-white">Gestión de Usuarios</h3>
          </div>
          <button
            onClick={() => setShowAdd(!showAdd)}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Nuevo Usuario
          </button>
        </div>

        {/* Add Form */}
        {showAdd && (
          <div className="mb-6 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
            <h4 className="text-sm font-medium text-white mb-3">Crear Nuevo Usuario</h4>
            <div className="space-y-3">
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Nombre"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
              <input
                type="email"
                value={newEmail}
                onChange={(e) => setNewEmail(e.target.value)}
                placeholder="Email"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
              <select
                value={newRole}
                onChange={(e) => setNewRole(e.target.value as any)}
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="viewer">Visualizador</option>
                <option value="operator">Operador</option>
                <option value="admin">Administrador</option>
              </select>
              <div className="flex gap-2">
                <button
                  onClick={handleAdd}
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
                >
                  Crear
                </button>
                <button
                  onClick={() => {
                    setShowAdd(false);
                    setNewName('');
                    setNewEmail('');
                  }}
                  className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
                >
                  Cancelar
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Users List */}
        <div className="space-y-3">
          {users.map((user) => (
            <div
              key={user.id}
              className={`p-4 rounded-lg border ${
                user.status === 'active'
                  ? 'bg-green-500/10 border-green-500/50'
                  : 'bg-gray-700/50 border-gray-600'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-semibold text-white">{user.name}</h4>
                    <span className={`px-2 py-0.5 rounded text-xs border ${getRoleColor(user.role)}`}>
                      {user.role}
                    </span>
                  </div>
                  <p className="text-sm text-gray-300 mb-1">{user.email}</p>
                  {user.lastLogin && (
                    <p className="text-xs text-gray-400">
                      Último acceso: {user.lastLogin.toLocaleString('es-ES')}
                    </p>
                  )}
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleToggleStatus(user.id)}
                    className={`px-3 py-1 text-sm rounded-lg transition-colors ${
                      user.status === 'active'
                        ? 'bg-yellow-600 hover:bg-yellow-700 text-white'
                        : 'bg-green-600 hover:bg-green-700 text-white'
                    }`}
                  >
                    {user.status === 'active' ? 'Desactivar' : 'Activar'}
                  </button>
                  <button
                    onClick={() => handleDelete(user.id)}
                    className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded-lg transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


