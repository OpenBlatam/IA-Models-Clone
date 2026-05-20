import { NextResponse } from 'next/server';
import crypto from 'crypto';

export async function GET() {
  const clientId = process.env.NEXT_PUBLIC_GITHUB_CLIENT_ID;
  const appUrl = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';

  if (!clientId) {
    return NextResponse.json(
      { error: 'GitHub Client ID no configurado' },
      { status: 500 }
    );
  }

  // Generar estado aleatorio para seguridad
  const state = crypto.randomBytes(32).toString('hex');

  // Construir URL de autorización de GitHub
  const redirectUri = `${appUrl}/github/callback`;
  const scope = 'repo user:email';
  
  const authUrl = `https://github.com/login/oauth/authorize?client_id=${clientId}&redirect_uri=${encodeURIComponent(redirectUri)}&scope=${encodeURIComponent(scope)}&state=${state}`;

  return NextResponse.json({
    auth_url: authUrl,
    state: state,
  });
}



