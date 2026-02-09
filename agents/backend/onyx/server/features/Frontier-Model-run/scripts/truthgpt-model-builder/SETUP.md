# Guía de Configuración - TruthGPT Model Builder

## 📋 Pasos de Instalación

### 1. Instalar Dependencias

```bash
npm install
```

### 2. Configurar Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto con las siguientes variables:

```env
GITHUB_TOKEN=ghp_tu_token_aqui
GITHUB_OWNER=tu_usuario_github
TRUTHGPT_API_PATH=../TruthGPT-main
```

#### Obtener Token de GitHub

1. Ve a: https://github.com/settings/tokens
2. Click en "Generate new token (classic)"
3. Selecciona los siguientes permisos:
   - ✅ `repo` (Full control of private repositories)
   - ✅ `workflow` (Update GitHub Action workflows)
4. Genera el token y cópialo
5. Pégalo en `.env` como `GITHUB_TOKEN`

### 3. Verificar Ruta de TruthGPT

Asegúrate de que la ruta a TruthGPT sea correcta. Por defecto es:
```
../TruthGPT-main
```

Si TruthGPT está en otra ubicación, actualiza `TRUTHGPT_API_PATH` en `.env`.

### 4. Ejecutar la Aplicación

```bash
npm run dev
```

La aplicación estará disponible en: http://localhost:3000

## 🔍 Verificación

1. Abre http://localhost:3000
2. Deberías ver la interfaz de chat
3. Escribe una descripción de modelo (ej: "Un modelo para análisis de sentimientos")
4. Click en "Crear Modelo"
5. El sistema debería:
   - Crear el modelo TruthGPT
   - Crear un repositorio en GitHub
   - Mostrar el progreso en tiempo real

## ⚠️ Solución de Problemas

### Error: "GITHUB_TOKEN no está configurado"

- Verifica que el archivo `.env` existe
- Verifica que `GITHUB_TOKEN` está definido
- Reinicia el servidor de desarrollo

### Error: "Cannot find module '../TruthGPT-main'"

- Verifica que TruthGPT está en la ruta correcta
- Ajusta `TRUTHGPT_API_PATH` en `.env`
- Asegúrate de que la ruta es relativa a la raíz del proyecto

### Error al crear repositorio en GitHub

- Verifica que el token tiene permisos `repo`
- Verifica que el nombre del modelo no está ya en uso
- Revisa los logs del servidor para más detalles

### El modelo no se crea

- Verifica que TruthGPT está correctamente instalado
- Verifica que Python está disponible
- Revisa los logs en la consola del servidor

## 📚 Recursos Adicionales

- [Documentación de Next.js](https://nextjs.org/docs)
- [GitHub API Documentation](https://docs.github.com/en/rest)
- [TruthGPT Documentation](../TruthGPT-main/README.md)


