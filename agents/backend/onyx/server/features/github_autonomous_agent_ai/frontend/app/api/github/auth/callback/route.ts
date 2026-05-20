import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const code = searchParams.get('code');
  const state = searchParams.get('state');

  if (!code) {
    return NextResponse.json(
      { error: 'No se proporcionó código de autorización' },
      { status: 400 }
    );
  }

  const clientId = process.env.NEXT_PUBLIC_GITHUB_CLIENT_ID;
  const clientSecret = process.env.GITHUB_CLIENT_SECRET;

  if (!clientId || !clientSecret) {
    return NextResponse.json(
      { error: 'Configuración de GitHub OAuth no encontrada' },
      { status: 500 }
    );
  }

  try {
    // Intercambiar código por token de acceso
    const tokenResponse = await fetch('https://github.com/login/oauth/access_token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({
        client_id: clientId,
        client_secret: clientSecret,
        code: code,
        state: state,
      }),
    });

    if (!tokenResponse.ok) {
      const errorText = await tokenResponse.text();
      console.error('Error obteniendo token:', errorText);
      return NextResponse.json(
        { error: 'Error al obtener token de acceso' },
        { status: tokenResponse.status }
      );
    }

    const tokenData = await tokenResponse.json();

    if (tokenData.error) {
      return NextResponse.json(
        { error: tokenData.error_description || tokenData.error },
        { status: 400 }
      );
    }

    const accessToken = tokenData.access_token;

    // Obtener información del usuario
    const userResponse = await fetch('https://api.github.com/user', {
      headers: {
        'Authorization': `Bearer ${accessToken}`,
        'Accept': 'application/vnd.github.v3+json',
      },
    });

    if (!userResponse.ok) {
      return NextResponse.json(
        { error: 'Error al obtener información del usuario' },
        { status: userResponse.status }
      );
    }

    const user = await userResponse.json();

    // Retornar token y usuario
    return NextResponse.json({
      success: true,
      access_token: accessToken,
      user: {
        login: user.login,
        name: user.name,
        avatar_url: user.avatar_url,
        email: user.email,
      },
    });
  } catch (error: any) {
    console.error('Error en callback OAuth:', error);
    return NextResponse.json(
      { error: error.message || 'Error al procesar autenticación' },
      { status: 500 }
    );
  }
}



