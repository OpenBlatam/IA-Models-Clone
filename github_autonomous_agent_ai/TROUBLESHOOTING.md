# Solución de Problemas - Autenticación GitHub

## Error: "Error al iniciar la autenticación"

### 1. Verificar Variables de Entorno

Asegúrate de que las siguientes variables estén configuradas en tu archivo `.env`:

```env
GITHUB_CLIENT_ID=tu_client_id_aqui
GITHUB_CLIENT_SECRET=tu_client_secret_aqui
GITHUB_REDIRECT_URI=http://localhost:3000/github/callback
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001
```

**Verificar en el backend:**
```bash
# En el directorio del backend
python -c "from config.settings import settings; print(f'CLIENT_ID: {settings.GITHUB_CLIENT_ID}'); print(f'REDIRECT_URI: {settings.GITHUB_REDIRECT_URI}')"
```

### 2. Verificar que el Servidor Backend Esté Corriendo

```bash
# Verificar que el servidor esté escuchando en el puerto correcto
curl http://localhost:8025/health
```

Deberías recibir: `{"status":"healthy"}`

### 3. Verificar CORS

Asegúrate de que `ALLOWED_ORIGINS` incluya la URL exacta del frontend:

```env
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### 4. Verificar la Configuración de GitHub OAuth App

1. Ve a https://github.com/settings/developers
2. Selecciona tu OAuth App
3. Verifica que:
   - **Authorization callback URL** sea exactamente: `http://localhost:3000/github/callback`
   - No tenga trailing slash
   - Coincida exactamente con `GITHUB_REDIRECT_URI`

### 5. Verificar Logs del Backend

Revisa los logs del servidor backend para ver errores específicos:

```bash
# Si estás corriendo con uvicorn directamente
# Los errores aparecerán en la consola

# Si estás usando un servicio, revisa los logs
tail -f logs/app.log  # o donde estén tus logs
```

### 6. Verificar la Consola del Navegador

Abre las herramientas de desarrollador (F12) y revisa:
- **Console**: Para errores de JavaScript
- **Network**: Para ver las peticiones HTTP y sus respuestas

### 7. Errores Comunes

#### "GITHUB_CLIENT_ID no configurado"
- **Solución**: Agrega `GITHUB_CLIENT_ID` a tu archivo `.env`
- Reinicia el servidor backend

#### "redirect_uri_mismatch"
- **Solución**: Verifica que `GITHUB_REDIRECT_URI` coincida exactamente con la URL configurada en GitHub
- No debe tener trailing slash
- Debe ser `http://localhost:3000/github/callback` (no `/api/github/auth/callback`)

#### "CORS error" o "Network Error"
- **Solución**: Verifica que `ALLOWED_ORIGINS` incluya la URL del frontend
- Verifica que el servidor backend esté corriendo
- Verifica que `NEXT_PUBLIC_API_URL` en el frontend apunte al backend correcto

#### El popup se cierra inmediatamente
- **Solución**: Verifica que los popups no estén bloqueados en tu navegador
- Revisa la consola del navegador para ver errores

#### "Error al comunicarse con GitHub"
- **Solución**: Verifica tu conexión a internet
- Verifica que las credenciales de GitHub sean correctas
- Revisa los logs del backend para más detalles

### 8. Probar el Endpoint Directamente

Puedes probar el endpoint de iniciar autenticación directamente:

```bash
curl http://localhost:8025/api/github/auth/initiate
```

Deberías recibir algo como:
```json
{
  "auth_url": "https://github.com/login/oauth/authorize?...",
  "state": "..."
}
```

Si recibes un error, el problema está en el backend.

### 9. Verificar que el Frontend Esté Configurado Correctamente

En el frontend, verifica que `NEXT_PUBLIC_API_URL` esté configurado:

```bash
# En el directorio del frontend
cat .env.local
# Debería tener:
# NEXT_PUBLIC_API_URL=http://localhost:8025
```

### 10. Reiniciar Todo

A veces simplemente necesitas reiniciar:

1. Detén el servidor backend
2. Detén el servidor frontend
3. Reinicia el backend
4. Reinicia el frontend
5. Limpia la caché del navegador (Ctrl+Shift+Delete)
6. Intenta de nuevo

## Obtener Ayuda

Si el problema persiste:

1. Revisa los logs completos del backend
2. Revisa la consola del navegador
3. Verifica que todas las variables de entorno estén configuradas
4. Verifica que la OAuth App en GitHub esté configurada correctamente



