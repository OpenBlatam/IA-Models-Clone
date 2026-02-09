# Quick Start Guide

## Instalación Rápida

1. **Instalar dependencias:**
```bash
cd frontend
npm install
```

2. **Configurar variables de entorno:**
Crea un archivo `.env.local` en la carpeta `frontend`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8010
```

3. **Iniciar el servidor de desarrollo:**
```bash
npm run dev
```

4. **Abrir en el navegador:**
```
http://localhost:3000
```

## Uso Básico

### Control del Robot

1. Ve a la pestaña **Control**
2. Ingresa las coordenadas X, Y, Z
3. Haz clic en **Mover**

### Chat

1. Ve a la pestaña **Chat**
2. Escribe comandos como:
   - `move to (0.5, 0.3, 0.2)`
   - `go home`
   - `stop`
   - `status`

### Monitoreo

- **Estado**: Ver estado del robot y sistema
- **Métricas**: Ver gráficos de CPU, memoria y rendimiento

## Solución de Problemas

### El frontend no se conecta al backend

1. Verifica que el backend esté corriendo en `http://localhost:8010`
2. Verifica la variable `NEXT_PUBLIC_API_URL` en `.env.local`
3. Revisa la consola del navegador para errores

### Errores de CORS

El backend debe tener CORS habilitado. Verifica la configuración del backend.

### WebSocket no conecta

1. Verifica que el backend soporte WebSocket
2. Revisa la URL del WebSocket en `lib/api/websocket.ts`

