'use client';

import { useState } from 'react';
import { LogIn, LogOut, User, Lock, Mail } from 'lucide-react';
import { toast } from '@/lib/utils/toast';

export default function AuthPanel() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    name: '',
  });

  const handleLogin = async () => {
    if (!formData.email || !formData.password) {
      toast.error('Por favor completa todos los campos');
      return;
    }

    // Simulate login
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setIsAuthenticated(true);
    toast.success('Inicio de sesión exitoso');
  };

  const handleRegister = async () => {
    if (!formData.email || !formData.password || !formData.name) {
      toast.error('Por favor completa todos los campos');
      return;
    }

    // Simulate registration
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setIsAuthenticated(true);
    toast.success('Registro exitoso');
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setFormData({ email: '', password: '', name: '' });
    toast.info('Sesión cerrada');
  };

  if (isAuthenticated) {
    return (
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-full bg-primary-600 flex items-center justify-center">
              <User className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="text-white font-semibold">{formData.name || 'Usuario'}</h3>
              <p className="text-sm text-gray-400">{formData.email}</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
          >
            <LogOut className="w-4 h-4" />
            Cerrar Sesión
          </button>
        </div>

        <div className="space-y-2 pt-4 border-t border-gray-700">
          <div className="flex items-center justify-between">
            <span className="text-gray-400">Rol</span>
            <span className="text-white font-medium">Administrador</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-400">Sesión activa</span>
            <span className="text-green-400">Activa</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
      <div className="flex items-center gap-2 mb-6">
        <Lock className="w-5 h-5 text-primary-400" />
        <h3 className="text-lg font-semibold text-white">
          {isLogin ? 'Iniciar Sesión' : 'Registrarse'}
        </h3>
      </div>

      <div className="space-y-4">
        {!isLogin && (
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Nombre</label>
            <div className="relative">
              <User className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                placeholder="Tu nombre"
              />
            </div>
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Email</label>
          <div className="relative">
            <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="email"
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              placeholder="tu@email.com"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Contraseña</label>
          <div className="relative">
            <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="password"
              value={formData.password}
              onChange={(e) => setFormData({ ...formData, password: e.target.value })}
              className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              placeholder="••••••••"
            />
          </div>
        </div>

        <button
          onClick={isLogin ? handleLogin : handleRegister}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors"
        >
          <LogIn className="w-4 h-4" />
          {isLogin ? 'Iniciar Sesión' : 'Registrarse'}
        </button>

        <button
          onClick={() => setIsLogin(!isLogin)}
          className="w-full text-sm text-gray-400 hover:text-gray-300 transition-colors"
        >
          {isLogin ? '¿No tienes cuenta? Regístrate' : '¿Ya tienes cuenta? Inicia sesión'}
        </button>
      </div>
    </div>
  );
}

