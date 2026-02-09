# Custom Hooks Documentation

Esta documentación describe todos los hooks personalizados disponibles en la aplicación.

## 📹 Video Generation Hooks

### `useGenerateVideo`
Genera un nuevo video desde un script.

```tsx
const generateVideo = useGenerateVideo();
await generateVideo.mutateAsync(request);
```

### `useVideoStatus`
Obtiene y monitorea el estado de un video. Hace polling automático si está procesando.

```tsx
const { data: video, isLoading } = useVideoStatus(videoId, enabled);
```

### `useDownloadVideo`
Descarga un video generado.

```tsx
const downloadVideo = useDownloadVideo();
await downloadVideo.mutateAsync({ videoId, onProgress });
```

### `useDeleteVideo`
Elimina un video.

```tsx
const deleteVideo = useDeleteVideo();
await deleteVideo.mutateAsync(videoId);
```

### `useUploadScript`
Sube un archivo de script.

```tsx
const uploadScript = useUploadScript();
await uploadScript.mutateAsync({ file, onProgress });
```

## 🎬 Video Actions Hooks

### `useShareVideo`
Comparte un video con otros usuarios.

```tsx
const shareVideo = useShareVideo();
await shareVideo.mutateAsync({ videoId, options });
```

### `useAddWatermark`
Agrega una marca de agua a un video.

```tsx
const addWatermark = useAddWatermark();
await addWatermark.mutateAsync({ videoId, options });
```

### `useTranscribeVideo`
Transcribe el audio de un video a texto.

```tsx
const transcribeVideo = useTranscribeVideo();
await transcribeVideo.mutateAsync({ videoId, language });
```

### `useAddKenBurnsEffect`
Agrega efecto Ken Burns a un video.

```tsx
const addKenBurns = useAddKenBurnsEffect();
await addKenBurns.mutateAsync({ videoId, options });
```

### `useExportVideo`
Exporta un video para múltiples plataformas.

```tsx
const exportVideo = useExportVideo();
await exportVideo.mutateAsync({ videoId, platforms });
```

### `useRegisterWebhook`
Registra un webhook para notificaciones.

```tsx
const registerWebhook = useRegisterWebhook();
await registerWebhook.mutateAsync({ videoId, webhookUrl });
```

## 📋 Template Hooks

### `useTemplates`
Lista todos los templates disponibles.

```tsx
const { data: templates } = useTemplates();
```

### `useTemplate`
Obtiene un template específico.

```tsx
const { data: template } = useTemplate(templateName, enabled);
```

### `useGenerateFromTemplate`
Genera un video usando un template.

```tsx
const generateFromTemplate = useGenerateFromTemplate();
await generateFromTemplate.mutateAsync({ templateName, scriptText, language });
```

### `useCustomTemplates`
Lista templates personalizados.

```tsx
const { data: customTemplates } = useCustomTemplates(userOnly);
```

### `useCreateCustomTemplate`
Crea un template personalizado.

```tsx
const createTemplate = useCreateCustomTemplate();
await createTemplate.mutateAsync({ name, description, config, isPublic });
```

## 📊 Analytics Hooks

### `useAnalytics`
Obtiene estadísticas y métricas.

```tsx
const { data: analytics } = useAnalytics();
```

### `useRecommendations`
Obtiene recomendaciones de IA para generación de videos.

```tsx
const { data: recommendations } = useRecommendations(
  scriptText,
  language,
  platform,
  contentType,
  enabled
);
```

### `useQuota`
Obtiene información de cuota del usuario.

```tsx
const { data: quota } = useQuota();
```

## 🔍 Search Hooks

### `useSearchVideos`
Busca videos con filtros.

```tsx
const { data: searchResults } = useSearchVideos(query, filters, limit, enabled);
```

### `useSearchSuggestions`
Obtiene sugerencias de búsqueda.

```tsx
const { data: suggestions } = useSearchSuggestions(query, limit, enabled);
```

### `useClearSearchHistory`
Limpia el historial de búsqueda.

```tsx
const clearHistory = useClearSearchHistory();
await clearHistory.mutateAsync();
```

## 📦 Batch Hooks

### `useBatchGenerate`
Genera múltiples videos en lote.

```tsx
const batchGenerate = useBatchGenerate();
await batchGenerate.mutateAsync(batchRequest);
```

### `useBatchStatus`
Obtiene el estado de un batch de videos.

```tsx
const { data: batchStatus } = useBatchStatus(videoIds, enabled);
```

## 🎵 Music Hooks

### `useMusicTracks`
Lista pistas de música disponibles.

```tsx
const { data: tracks } = useMusicTracks(style);
```

## 💬 Feedback Hooks

### `useSubmitFeedback`
Envía feedback sobre un video.

```tsx
const submitFeedback = useSubmitFeedback();
await submitFeedback.mutateAsync({ videoId, feedback });
```

### `useVideoFeedback`
Obtiene feedback de un video.

```tsx
const { data: feedback } = useVideoFeedback(videoId, enabled);
```

## ⏰ Scheduled Hooks

### `useScheduleVideo`
Programa un video para generación futura.

```tsx
const scheduleVideo = useScheduleVideo();
await scheduleVideo.mutateAsync({ videoId, options });
```

### `useScheduledVideos`
Lista videos programados.

```tsx
const { data: scheduled } = useScheduledVideos(videoId, status, enabled);
```

### `useCancelScheduledJob`
Cancela un trabajo programado.

```tsx
const cancelJob = useCancelScheduledJob();
await cancelJob.mutateAsync(jobId);
```

## 🎨 UI Hooks

### `useTheme`
Maneja el tema de la aplicación (light/dark/auto).

```tsx
const { theme, isDark, setTheme } = useTheme();
```

### `useKeyboard`
Monitorea el estado del teclado.

```tsx
const { isKeyboardVisible, keyboardHeight } = useKeyboard();
```

### `useScreenDimensions`
Obtiene dimensiones de pantalla y breakpoints.

```tsx
const { width, height, isTablet, isSmallScreen } = useScreenDimensions();
```

### `useHapticFeedback`
Proporciona feedback háptico.

```tsx
const haptics = useHapticFeedback();
haptics.success();
haptics.error();
```

### `useDebounce`
Debounce de valores.

```tsx
const debouncedValue = useDebounce(value, 500);
```

### `useThrottle`
Throttle de valores.

```tsx
const throttledValue = useThrottle(value, 1000);
```

### `useToggle`
Toggle de valores booleanos.

```tsx
const { value, toggle, setTrue, setFalse } = useToggle();
```

### `usePrevious`
Obtiene el valor anterior.

```tsx
const previousValue = usePrevious(currentValue);
```

### `useInterval`
Ejecuta una función en intervalos.

```tsx
useInterval(() => {
  // Do something
}, 1000);
```

### `useTimeout`
Ejecuta una función después de un delay.

```tsx
useTimeout(() => {
  // Do something
}, 5000);
```

## 📁 File Operations Hooks

### `useFilePicker`
Selecciona un archivo del dispositivo.

```tsx
const { pickFile, isPicking } = useFilePicker();
const file = await pickFile({ type: ['text/*'] });
```

### `useFileDownload`
Descarga un archivo.

```tsx
const { downloadFile, isDownloading, progress } = useFileDownload();
const uri = await downloadFile(url, filename, onProgress);
```

### `useFileShare`
Comparte un archivo.

```tsx
const { shareFile, isSharing } = useFileShare();
await shareFile(uri, { mimeType: 'video/mp4' });
```

### `useFileDelete`
Elimina un archivo.

```tsx
const { deleteFile } = useFileDelete();
await deleteFile(uri);
```

### `useFileInfo`
Obtiene información de un archivo.

```tsx
const { getFileInfo } = useFileInfo();
const info = await getFileInfo(uri);
```

## 🔔 Notification Hooks

### `useNotificationPermissions`
Maneja permisos de notificaciones.

```tsx
const { hasPermission, requestPermissions } = useNotificationPermissions();
```

### `usePushNotifications`
Envía y programa notificaciones.

```tsx
const { sendLocalNotification, scheduleNotification } = usePushNotifications();
await sendLocalNotification('Title', 'Body', { data });
```

## 🌐 Network Hooks

### `useNetworkStatus`
Monitorea el estado de la conexión de red.

```tsx
const { isConnected, connectionType, isOffline } = useNetworkStatus();
```

### `useNetworkRequest`
Ejecuta requests con manejo de red.

```tsx
const { execute, isLoading, error } = useNetworkRequest(
  () => api.getData(),
  { retryOnNetworkError: true }
);
```

## ✅ Validation Hooks

### `useFormValidation`
Valida formularios completos.

```tsx
const { validate, errors, getFieldError, setFieldTouched } = useFormValidation(schema);
const isValid = validate(formData);
```

### `useFieldValidation`
Valida campos individuales.

```tsx
const { validate, error, markAsTouched } = useFieldValidation(schema, 'email');
const isValid = validate(emailValue);
```

## ⚡ Performance Hooks

### `usePerformanceMonitor`
Monitorea el rendimiento de renders.

```tsx
const { renderCount } = usePerformanceMonitor();
```

### `useDeferredCallback`
Ejecuta callbacks de forma diferida.

```tsx
const deferredCallback = useDeferredCallback(() => {
  // Heavy operation
}, 100);
```

### `useInteractionCallback`
Ejecuta callbacks después de interacciones.

```tsx
const interactionCallback = useInteractionCallback(() => {
  // Do something after interactions
});
```

### `useRenderCount`
Cuenta los renders de un componente.

```tsx
const renderCount = useRenderCount('MyComponent');
```

### `useMemoizedCallback`
Memoiza callbacks con dependencias.

```tsx
const memoizedCallback = useMemoizedCallback(
  (value) => {
    // Do something
  },
  [dependency]
);
```

## 📝 Video Versions Hooks

### `useVideoVersions`
Obtiene todas las versiones de un video.

```tsx
const { data: versions } = useVideoVersions(videoId, enabled);
```

### `useVideoVersion`
Obtiene una versión específica.

```tsx
const { data: version } = useVideoVersion(videoId, versionNumber, enabled);
```

### `useCompareVersions`
Compara dos versiones de un video.

```tsx
const compareVersions = useCompareVersions();
await compareVersions.mutateAsync({ videoId, version1, version2 });
```

## 🎯 Mejores Prácticas

1. **Siempre maneja estados de carga y error**:
```tsx
const { data, isLoading, error } = useVideoStatus(videoId);
if (isLoading) return <Loading />;
if (error) return <Error />;
```

2. **Usa `enabled` para controlar queries**:
```tsx
const { data } = useVideoStatus(videoId, !!videoId);
```

3. **Memoiza callbacks cuando sea necesario**:
```tsx
const handleSubmit = useCallback(() => {
  // ...
}, [dependencies]);
```

4. **Usa debounce para búsquedas**:
```tsx
const debouncedQuery = useDebounce(query, 300);
const { data } = useSearchVideos(debouncedQuery);
```

5. **Maneja permisos antes de usar hooks que los requieren**:
```tsx
const { hasPermission, requestPermissions } = useNotificationPermissions();
if (!hasPermission) {
  await requestPermissions();
}
```

## 📚 Ejemplos Completos

### Ejemplo: Generar Video con Validación

```tsx
function VideoGenerationForm() {
  const generateVideo = useGenerateVideo();
  const { validate, errors, getFieldError } = useFormValidation(videoGenerationSchema);
  const [formData, setFormData] = useState({ script: '', style: 'realistic' });

  const handleSubmit = async () => {
    if (validate(formData)) {
      await generateVideo.mutateAsync({
        script: { text: formData.script },
        video_config: { style: formData.style },
      });
    }
  };

  return (
    <View>
      <Input
        value={formData.script}
        onChangeText={(text) => setFormData({ ...formData, script: text })}
        error={getFieldError('script')}
      />
      <Button onPress={handleSubmit} loading={generateVideo.isPending} />
    </View>
  );
}
```

### Ejemplo: Monitoreo de Video con Notificaciones

```tsx
function VideoMonitor({ videoId }: { videoId: string }) {
  const { data: video } = useVideoStatus(videoId);
  const { sendLocalNotification } = usePushNotifications();

  useEffect(() => {
    if (video?.status === 'completed') {
      sendLocalNotification('Video Ready', 'Your video has been generated!', {
        videoId,
      });
    }
  }, [video?.status, videoId, sendLocalNotification]);

  return <VideoProgress video={video} />;
}
```

---

**Nota**: Todos los hooks están optimizados para React Native y Expo, y siguen las mejores prácticas de performance y UX.


