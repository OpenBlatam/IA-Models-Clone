'use client';

import { useEffect, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';

function CallbackContent() {
  const searchParams = useSearchParams();
  const router = useRouter();

  useEffect(() => {
    const code = searchParams.get('code');
    const state = searchParams.get('state');
    const error = searchParams.get('error');

    if (error) {
      // Enviar mensaje de error al padre si es popup
      if (window.opener) {
        window.opener.postMessage(
          { type: 'github-auth-error', error },
          window.location.origin
        );
        window.close();
      } else {
        // Si no es popup, redirigir a la página principal con error
        router.push(`/?error=${encodeURIComponent(error)}`);
      }
      return;
    }

    if (code && state) {
      // Si es popup, enviar código y estado al padre
      if (window.opener) {
        window.opener.postMessage(
          { type: 'github-auth-success', code, state },
          window.location.origin
        );
        window.close();
      } else {
        // Si no es popup, procesar directamente
        handleCallback(code, state);
      }
    }
  }, [searchParams, router]);

  const handleCallback = async (code: string, state: string) => {
    try {
      const response = await fetch(`/api/github/auth/callback?code=${code}&state=${state}`);
      const data = await response.json();
      
      if (data.success) {
        // Guardar token y usuario en localStorage
        if (typeof window !== 'undefined') {
          localStorage.setItem('github_access_token', data.access_token);
          localStorage.setItem('github_user', JSON.stringify(data.user));
        }
        router.push('/?auth=success');
      } else {
        router.push(`/?error=${encodeURIComponent(data.error || 'Error de autenticación')}`);
      }
    } catch (error) {
      router.push(`/?error=${encodeURIComponent('Error al procesar autenticación')}`);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
        <p className="text-gray-600">Procesando autenticación...</p>
      </div>
    </div>
  );
}

export default function GithubCallback() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    }>
      <CallbackContent />
    </Suspense>
  );
}

