# Configuración de GitHub OAuth

Para que la autenticación con GitHub funcione correctamente, necesitas configurar la URL de redirección en tu aplicación OAuth de GitHub.

## Pasos

1. Ve a [GitHub Developer Settings](https://github.com/settings/developers)
2. Selecciona tu OAuth App (o crea una nueva)
3. En "Authorization callback URL", agrega:
   - Para desarrollo local: `http://localhost:3000/github/callback`
   - Para producción: `https://tu-dominio.com/github/callback`

## Credenciales Actuales

- **Client ID**: `Ov23liSy9XyIj3dD0dQc`
- **Client Secret**: `6ed948f00e7662bbba0eacd356e60747dd12f08f`

Estas credenciales ya están configuradas en `.env.example` y deben estar en tu `.env.local`.

## Nota de Seguridad

El `GITHUB_CLIENT_SECRET` solo se usa en las API routes del servidor de Next.js (archivos en `app/api/`), nunca se expone al cliente del navegador.



