'use client';

import { useState, useEffect } from 'react';
import { githubAPI, GitHubUser } from '../lib/github-api';

interface GithubAuthProps {
  onAuthSuccess: (user: GitHubUser) => void;
  onAuthError: (error: string) => void;
}

export default function GithubAuth({ onAuthSuccess, onAuthError }: GithubAuthProps) {
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
    try {
      setAuthenticating(true);
      const token = tokenInput.trim() || process.env.NEXT_PUBLIC_GITHUB_DEFAULT_TOKEN || '';
      const { success, user: authUser } = await githubAPI.authenticateWithToken(token);

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

  const handleQuickConnect = async () => {
    try {
      setAuthenticating(true);
      const defaultToken = process.env.NEXT_PUBLIC_GITHUB_DEFAULT_TOKEN || '';
      if (!defaultToken) {
        onAuthError('No hay token configurado. Usa "Usar mi propio token" para conectar con tu Personal Access Token de GitHub.');
        setAuthenticating(false);
        return;
      }
      const { success, user: authUser } = await githubAPI.authenticateWithToken(defaultToken);

      if (success && authUser) {
        setUser(authUser);
        onAuthSuccess(authUser);
      } else {
        onAuthError('Error al conectar con el token por defecto');
      }
    } catch (error: any) {
      onAuthError(error.message || 'Error al conectar');
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
      sessionStorage.setItem('github_oauth_state', state);

      // Abrir ventana de OAuth
      const width = 600;
      const height = 700;
      const left = window.screen.width / 2 - width / 2;
      const top = window.screen.height / 2 - height / 2;

      const popup = window.open(
        auth_url,
        'GitHub Auth',
        `width=${width},height=${height},left=${left},top=${top}`
      );

      if (!popup) {
        throw new Error('No se pudo abrir la ventana de autenticación. Verifica que los popups no estén bloqueados.');
      }

      // Escuchar mensaje del popup
      const messageListener = async (event: MessageEvent) => {
        if (event.origin !== window.location.origin) return;

        if (event.data.type === 'github-auth-success') {
          const { code, state: callbackState } = event.data;
          const savedState = sessionStorage.getItem('github_oauth_state');

          // Validar estado
          if (callbackState !== savedState) {
            onAuthError('Error de seguridad: estado no coincide');
            window.removeEventListener('message', messageListener);
            setAuthenticating(false);
            return;
          }

          try {
            const { success, user: authUser } = await githubAPI.handleCallback(code, callbackState);
            if (success && authUser) {
              setUser(authUser);
              onAuthSuccess(authUser);
              sessionStorage.removeItem('github_oauth_state');
            } else {
              onAuthError('Error al autenticar con GitHub');
            }
          } catch (error) {
            onAuthError('Error al procesar la autenticación');
          } finally {
            window.removeEventListener('message', messageListener);
            setAuthenticating(false);
          }
        } else if (event.data.type === 'github-auth-error') {
          onAuthError(event.data.error || 'Error al autenticar con GitHub');
          window.removeEventListener('message', messageListener);
          setAuthenticating(false);
          sessionStorage.removeItem('github_oauth_state');
        }
      };

      window.addEventListener('message', messageListener);

      // Verificar si el popup fue cerrado
      const checkClosed = setInterval(() => {
        if (popup?.closed) {
          clearInterval(checkClosed);
          window.removeEventListener('message', messageListener);
          setAuthenticating(false);
          sessionStorage.removeItem('github_oauth_state');
        }
      }, 1000);
    } catch (error: any) {
      console.error('Error initiating auth:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Error al iniciar la autenticación';
      onAuthError(errorMessage);
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
      <div className="flex items-center justify-center p-4">
        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (user) {
    return (
      <div className="flex items-center gap-4 p-4 bg-white rounded-lg shadow">
        <img
          src={user.avatar_url}
          alt={user.login}
          className="w-10 h-10 rounded-full"
        />
        <div className="flex-1">
          <div className="font-semibold">{user.name || user.login}</div>
          <div className="text-sm text-gray-500">@{user.login}</div>
        </div>
        <button
          onClick={handleLogout}
          className="px-4 py-2 text-sm bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
        >
          Cerrar sesión
        </button>
      </div>
    );
  }

  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <div className="text-center">
        <h3 className="text-lg font-semibold mb-2">Conectar con GitHub</h3>
        <p className="text-sm text-gray-600 mb-4">
          Conecta tu cuenta de GitHub usando un token de acceso
        </p>

        {/* Botón de conexión rápida con token por defecto */}
        <button
          onClick={handleQuickConnect}
          disabled={authenticating}
          className="w-full mb-3 px-6 py-2 bg-gray-900 text-white rounded-lg hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {authenticating ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              <span>Conectando...</span>
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
              </svg>
              <span>Conectar con Token Predefinido</span>
            </>
          )}
        </button>

        {/* Opción para usar token personalizado */}
        <div className="mt-4">
          <button
            onClick={() => setUseToken(!useToken)}
            className="text-sm text-blue-600 hover:text-blue-800 mb-2"
          >
            {useToken ? 'Ocultar' : 'Usar mi propio token'}
          </button>

          {useToken && (
            <div className="space-y-2">
              <input
                type="password"
                value={tokenInput}
                onChange={(e) => setTokenInput(e.target.value)}
                placeholder="ghp_xxxxxxxxxxxxx"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                onClick={handleTokenAuth}
                disabled={authenticating}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
              >
                {authenticating ? 'Conectando...' : 'Conectar con Token'}
              </button>
            </div>
          )}
        </div>

        {/* Opción OAuth (alternativa) */}
        <div className="mt-4 pt-4 border-t border-gray-200">
          <button
            onClick={handleLogin}
            disabled={authenticating}
            className="text-sm text-gray-600 hover:text-gray-800"
          >
            O usar OAuth (método alternativo)
          </button>
        </div>
      </div>
    </div>
  );
}

