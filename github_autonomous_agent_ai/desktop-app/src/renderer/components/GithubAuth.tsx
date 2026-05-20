import React, { useState, useEffect } from 'react';
import { githubAPI, GitHubUser } from '../lib/github-api';
import { Button } from './ui/Button';
import { Input } from './ui/Input';

interface GithubAuthProps {
  onAuthSuccess: (user: GitHubUser) => void;
  onAuthError: (error: string) => void;
}

export const GithubAuth: React.FC<GithubAuthProps> = ({ onAuthSuccess, onAuthError }) => {
  const [user, setUser] = useState<GitHubUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [authenticating, setAuthenticating] = useState(false);
  const [tokenInput, setTokenInput] = useState('');
  const [useToken, setUseToken] = useState(false);

  useEffect(() => {
    checkAuthStatus();
  }, []);

  const checkAuthStatus = async () => {
    try {
      setLoading(true);
      const { authenticated, user: authUser } = await githubAPI.checkAuth();
      if (authenticated && authUser) {
        setUser(authUser);
        onAuthSuccess(authUser);
      }
    } catch (error) {
      console.error('Error checking auth status:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleTokenAuth = async () => {
    if (!tokenInput.trim()) {
      onAuthError('Por favor, ingresa un token válido');
      return;
    }

    try {
      setAuthenticating(true);
      const { success, user: authUser } = await githubAPI.authenticateWithToken(tokenInput.trim());
      
      if (success && authUser) {
        setUser(authUser);
        onAuthSuccess(authUser);
        setTokenInput('');
        setUseToken(false);
      } else {
        onAuthError('Token inválido. Por favor, verifica tu token de GitHub.');
      }
    } catch (error: any) {
      onAuthError(error.message || 'Error al autenticar con token');
    } finally {
      setAuthenticating(false);
    }
  };

  const handleLogin = async () => {
    try {
      setAuthenticating(true);
      const response = await githubAPI.initiateAuth();
      
      if (!response || !response.auth_url) {
        throw new Error('No se recibió URL de autorización del servidor');
      }
      
      const { auth_url, state } = response;
      
      // Guardar el estado para validación posterior
      if (typeof window !== 'undefined') {
        sessionStorage.setItem('github_oauth_state', state);
      }
      
      // En Electron, podemos abrir la URL en el navegador externo
      if (window.electronAPI) {
        // Usar shell para abrir URL externa
        window.open(auth_url, '_blank');
        onAuthError('Por favor, completa la autenticación en el navegador y luego ingresa el código aquí.');
      } else {
        // Fallback para desarrollo web
        window.open(auth_url, 'GitHub Auth', 'width=600,height=700');
      }
    } catch (error: any) {
      console.error('Error initiating auth:', error);
      onAuthError(error.message || 'Error al iniciar la autenticación');
    } finally {
      setAuthenticating(false);
    }
  };

  const handleLogout = async () => {
    try {
      await githubAPI.logout();
      setUser(null);
    } catch (error) {
      console.error('Error logging out:', error);
    }
  };

  if (loading) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="flex items-center justify-center p-4">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-black"></div>
        </div>
      </div>
    );
  }

  if (user) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="flex items-center gap-4">
          <img
            src={user.avatar_url}
            alt={user.login}
            className="w-10 h-10 rounded-full"
          />
          <div className="flex-1">
            <div className="font-medium text-black">{user.name || user.login}</div>
            <div className="text-sm text-gray-500">@{user.login}</div>
          </div>
          <Button variant="secondary" size="sm" onClick={handleLogout}>
            Cerrar sesión
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6">
      <div className="text-center">
        <h3 className="text-black text-xl font-normal mb-1">Conectar con GitHub</h3>
        <p className="text-sm text-gray-500 mb-4">
          Conecta tu cuenta de GitHub usando un token de acceso
        </p>
          
          {useToken ? (
            <div className="space-y-3">
              <Input
                type="password"
                value={tokenInput}
                onChange={(e) => setTokenInput(e.target.value)}
                placeholder="ghp_xxxxxxxxxxxxx"
                label="Token de GitHub"
                fullWidth
              />
              <div className="flex gap-2">
                <Button
                  variant="primary"
                  fullWidth
                  onClick={handleTokenAuth}
                  isLoading={authenticating}
                >
                  Conectar
                </Button>
                <Button
                  variant="ghost"
                  onClick={() => {
                    setUseToken(false);
                    setTokenInput('');
                  }}
                >
                  Cancelar
                </Button>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <Button
                variant="primary"
                fullWidth
                onClick={() => setUseToken(true)}
              >
                Usar Token
              </Button>
              <Button
                variant="outline"
                fullWidth
                onClick={handleLogin}
                isLoading={authenticating}
              >
                OAuth (Navegador)
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};


