# Guía de Autenticación con Google

## 🔐 Configuración

### 1. Crear Proyecto en Google Cloud Console

1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
2. Crea un nuevo proyecto o selecciona uno existente
3. Habilita la API de Google+ o Google Identity
4. Ve a "Credenciales" > "Crear credenciales" > "ID de cliente OAuth 2.0"
5. Configura:
   - Tipo: Aplicación web
   - Orígenes autorizados: `http://localhost:3000` (desarrollo)
   - URI de redirección autorizados: `http://localhost:3000/api/auth/callback/google`

### 2. Variables de Entorno

Crea un archivo `.env.local` en la raíz del proyecto:

```env
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=tu-clave-secreta-aqui
GOOGLE_CLIENT_ID=tu-google-client-id
GOOGLE_CLIENT_SECRET=tu-google-client-secret
```

Para generar `NEXTAUTH_SECRET`:
```bash
openssl rand -base64 32
```

### 3. Instalación

```bash
npm install
```

## 🚀 Uso

### Página de Login

La página de login está en `/login` y se muestra automáticamente cuando el usuario no está autenticado.

### Proteger Rutas

Las rutas están protegidas automáticamente por el middleware. Si el usuario no está autenticado, se redirige a `/login`.

### Hook useAuth

```typescript
import { useAuth } from '@/hooks/useAuth';

const { user, isAuthenticated, isLoading, signOut } = useAuth();
```

### Componente ProtectedRoute

```typescript
import { ProtectedRoute } from '@/components/auth/ProtectedRoute';

<ProtectedRoute>
  <YourComponent />
</ProtectedRoute>
```

### Información del Usuario

```typescript
const { user } = useAuth();

// user.name
// user.email
// user.image
// user.id
```

## 📁 Archivos Creados

1. `app/api/auth/[...nextauth]/route.ts` - API route de NextAuth
2. `app/[locale]/login/page.tsx` - Página de login
3. `components/auth/SessionProvider.tsx` - Provider de sesión
4. `components/auth/ProtectedRoute.tsx` - Componente para proteger rutas
5. `components/auth/UserMenu.tsx` - Menú de usuario
6. `hooks/useAuth.ts` - Hook para autenticación
7. `types/next-auth.d.ts` - Tipos TypeScript
8. `middleware.ts` - Middleware de autenticación

## 🎨 Características

- ✅ Login con Google OAuth
- ✅ Protección automática de rutas
- ✅ Sesión persistente (30 días)
- ✅ Menú de usuario con logout
- ✅ Redirección automática
- ✅ Estados de carga
- ✅ Manejo de errores
- ✅ Soporte multiidioma
- ✅ TypeScript completo

## 🔒 Seguridad

- JWT tokens seguros
- Sesiones en servidor
- CSRF protection
- Secure cookies
- HTTPS en producción

## 🌐 Producción

Para producción, actualiza:

1. Variables de entorno con valores reales
2. `NEXTAUTH_URL` con tu dominio
3. Google OAuth con URLs de producción
4. `GOOGLE_CLIENT_ID` y `GOOGLE_CLIENT_SECRET` de producción

## 🐛 Troubleshooting

### Error: "NEXTAUTH_SECRET is not set"

Agrega `NEXTAUTH_SECRET` a tu `.env.local`

### Error: "Invalid credentials"

Verifica que `GOOGLE_CLIENT_ID` y `GOOGLE_CLIENT_SECRET` sean correctos

### Redirect loop

Verifica que `NEXTAUTH_URL` coincida con tu dominio

### Session not persisting

Verifica que las cookies estén habilitadas y que `NEXTAUTH_SECRET` esté configurado



