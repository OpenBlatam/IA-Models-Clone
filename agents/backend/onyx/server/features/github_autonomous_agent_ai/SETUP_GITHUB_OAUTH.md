# Configuración de GitHub OAuth

Para que la autenticación con GitHub funcione, necesitas configurar una aplicación OAuth en GitHub.

## Pasos para configurar GitHub OAuth

1. **Ir a GitHub Settings**
   - Ve a https://github.com/settings/developers
   - O directamente a https://github.com/settings/applications/new

2. **Crear una nueva OAuth App**
   - Click en "New OAuth App"
   - Completa el formulario:
     - **Application name**: GitHub Autonomous Agent AI (o el nombre que prefieras)
     - **Homepage URL**: `http://localhost:3000` (o tu URL de producción)
     - **Authorization callback URL**: `http://localhost:3000/github/callback`
     - **Application description**: (opcional)

3. **Obtener credenciales**
   - Después de crear la app, GitHub te dará:
     - **Client ID**
     - **Client Secret** (haz click en "Generate a new client secret" si no lo ves)

4. **Configurar variables de entorno**

   Crea o actualiza tu archivo `.env` en la raíz del proyecto backend:

   ```env
   # GitHub OAuth
   GITHUB_CLIENT_ID=Ov23liSy9XyIj3dD0dQc
   GITHUB_CLIENT_SECRET=6ed948f00e7662bbba0eacd356e60747dd12f08f
   GITHUB_REDIRECT_URI=http://localhost:8025/api/github/auth/callback
   
   # CORS (para desarrollo local)
   ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000
   
   # DeepSeek LLM
   DEEPSEEK_API_KEY=sk-ae1c47feaa3e483b85a936430d1f494a
   LLM_ENABLED=true
   ```
   
   **Nota importante**: Asegúrate de que la **Authorization callback URL** en GitHub sea exactamente:
   `http://localhost:8025/api/github/auth/callback`

5. **Para producción**

   Cuando despliegues a producción, actualiza:
   - La **Authorization callback URL** en GitHub con tu URL de producción
   - Las variables de entorno con las URLs correctas:
     ```env
     GITHUB_REDIRECT_URI=https://tu-dominio.com/github/callback
     ALLOWED_ORIGINS=https://tu-dominio.com
     ```

## Verificar la configuración

1. Reinicia el servidor backend después de agregar las variables de entorno
2. Abre el frontend en `http://localhost:3000`
3. Haz click en "Conectar con GitHub"
4. Deberías ser redirigido a GitHub para autorizar la aplicación
5. Después de autorizar, serás redirigido de vuelta y estarás autenticado

## Solución de problemas

### Error: "GITHUB_CLIENT_ID no configurado"
- Verifica que las variables de entorno estén configuradas correctamente
- Asegúrate de reiniciar el servidor después de cambiar las variables

### Error: "redirect_uri_mismatch"
- Verifica que la URL en `GITHUB_REDIRECT_URI` coincida exactamente con la configurada en GitHub
- No debe tener trailing slash ni diferencias en http/https

### Error: "No autenticado" al obtener repositorios
- Verifica que las cookies estén habilitadas en tu navegador
- Asegúrate de que CORS esté configurado correctamente en el backend

### El popup se cierra inmediatamente
- Verifica la consola del navegador para ver errores
- Asegúrate de que el frontend esté corriendo en el puerto correcto (3000 por defecto)

