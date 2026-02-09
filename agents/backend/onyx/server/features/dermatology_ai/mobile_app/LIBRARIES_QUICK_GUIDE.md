# 🚀 Guía Rápida de Librerías - Dermatology AI Mobile

## 📋 Uso Rápido de Librerías Principales

### 1. **State Management - Redux Toolkit**

```typescript
// store/store.ts
import { configureStore } from '@reduxjs/toolkit';
import { persistStore, persistReducer } from 'redux-persist';
import AsyncStorage from '@react-native-async-storage/async-storage';

const persistConfig = {
  key: 'root',
  storage: AsyncStorage,
};

const store = configureStore({
  reducer: {
    // tus reducers aquí
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: false,
    }),
});

export const persistor = persistStore(store);
export default store;
```

### 2. **Data Fetching - React Query**

```typescript
// hooks/useAnalysis.ts
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

export const useAnalysis = (imageId: string) => {
  return useQuery({
    queryKey: ['analysis', imageId],
    queryFn: async () => {
      const { data } = await axios.get(`/api/analysis/${imageId}`);
      return data;
    },
    enabled: !!imageId,
  });
};
```

### 3. **Formularios - React Hook Form + Zod**

```typescript
// components/UserForm.tsx
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

const schema = z.object({
  name: z.string().min(2),
  email: z.string().email(),
});

export const UserForm = () => {
  const { control, handleSubmit } = useForm({
    resolver: zodResolver(schema),
  });

  const onSubmit = (data) => {
    console.log(data);
  };

  return (
    // tu formulario aquí
  );
};
```

### 4. **Navegación - React Navigation**

```typescript
// navigation/AppNavigator.tsx
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

const Stack = createStackNavigator();
const Tabs = createBottomTabNavigator();

export const AppNavigator = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Home" component={HomeScreen} />
        {/* más pantallas */}
      </Stack.Navigator>
    </NavigationContainer>
  );
};
```

### 5. **Imágenes Optimizadas - Fast Image**

```typescript
import FastImage from 'react-native-fast-image';

<FastImage
  source={{
    uri: 'https://example.com/image.jpg',
    priority: FastImage.priority.normal,
  }}
  style={{ width: 200, height: 200 }}
  resizeMode={FastImage.resizeMode.contain}
/>
```

### 6. **Notificaciones - Expo Notifications**

```typescript
import * as Notifications from 'expo-notifications';

// Solicitar permisos
const { status } = await Notifications.requestPermissionsAsync();

// Enviar notificación local
await Notifications.scheduleNotificationAsync({
  content: {
    title: "Análisis completado",
    body: 'Tu análisis de piel está listo',
  },
  trigger: null, // inmediato
});
```

### 7. **Almacenamiento Seguro - MMKV**

```typescript
import { MMKV } from 'react-native-mmkv';

const storage = new MMKV();

// Guardar
storage.set('user.token', 'abc123');

// Leer
const token = storage.getString('user.token');

// Eliminar
storage.delete('user.token');
```

### 8. **Autenticación Biométrica**

```typescript
import * as LocalAuthentication from 'expo-local-authentication';

const authenticate = async () => {
  const hasHardware = await LocalAuthentication.hasHardwareAsync();
  const isEnrolled = await LocalAuthentication.isEnrolledAsync();

  if (hasHardware && isEnrolled) {
    const result = await LocalAuthentication.authenticateAsync({
      promptMessage: 'Autentícate para continuar',
    });
    
    if (result.success) {
      // Autenticación exitosa
    }
  }
};
```

### 9. **Gráficos - Victory Native**

```typescript
import { VictoryChart, VictoryLine, VictoryAxis } from 'victory-native';

<VictoryChart>
  <VictoryAxis />
  <VictoryAxis dependentAxis />
  <VictoryLine
    data={[
      { x: 1, y: 2 },
      { x: 2, y: 3 },
      { x: 3, y: 5 },
    ]}
  />
</VictoryChart>
```

### 10. **Modales - React Native Modal**

```typescript
import Modal from 'react-native-modal';

<Modal
  isVisible={isVisible}
  onBackdropPress={() => setIsVisible(false)}
  animationIn="slideInUp"
  animationOut="slideOutDown"
>
  <View>
    {/* contenido del modal */}
  </View>
</Modal>
```

### 11. **Bottom Sheet - Gorhom Bottom Sheet**

```typescript
import BottomSheet from '@gorhom/bottom-sheet';

const snapPoints = ['25%', '50%', '90%'];

<BottomSheet
  ref={bottomSheetRef}
  index={1}
  snapPoints={snapPoints}
  enablePanDownToClose
>
  <View>
    {/* contenido */}
  </View>
</BottomSheet>
```

### 12. **Toasts - React Native Toast Message**

```typescript
import Toast from 'react-native-toast-message';

// En tu componente
Toast.show({
  type: 'success',
  text1: 'Éxito',
  text2: 'Operación completada',
});

// En App.tsx
<Toast />
```

### 13. **Calendarios - React Native Calendars**

```typescript
import { Calendar } from 'react-native-calendars';

<Calendar
  onDayPress={(day) => {
    console.log('selected day', day);
  }}
  markedDates={{
    '2024-01-15': { selected: true, marked: true },
  }}
/>
```

### 14. **Cámara - Expo Camera**

```typescript
import { CameraView, useCameraPermissions } from 'expo-camera';

const [permission, requestPermission] = useCameraPermissions();

if (!permission) {
  return <View />;
}

if (!permission.granted) {
  return (
    <View>
      <Text>Necesitamos permiso para usar la cámara</Text>
      <Button onPress={requestPermission} title="Conceder permiso" />
    </View>
  );
}

<CameraView
  style={{ flex: 1 }}
  facing="back"
  onBarcodeScanned={handleBarcodeScanned}
/>
```

### 15. **Animaciones - Lottie**

```typescript
import LottieView from 'lottie-react-native';

<LottieView
  source={require('./animation.json')}
  autoPlay
  loop
  style={{ width: 200, height: 200 }}
/>
```

### 16. **Internacionalización - i18next**

```typescript
// i18n/config.ts
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

i18n.use(initReactI18next).init({
  resources: {
    en: { translation: { welcome: 'Welcome' } },
    es: { translation: { welcome: 'Bienvenido' } },
  },
  lng: 'es',
});

// En componentes
import { useTranslation } from 'react-i18next';

const { t } = useTranslation();
<Text>{t('welcome')}</Text>
```

### 17. **Error Tracking - Sentry**

```typescript
import * as Sentry from '@sentry/react-native';

Sentry.init({
  dsn: 'YOUR_DSN',
  enableInExpoDevelopment: true,
  debug: true,
});

// Capturar error
Sentry.captureException(error);
```

### 18. **Deep Linking - Expo Linking**

```typescript
import * as Linking from 'expo-linking';

// Obtener URL inicial
const initialUrl = await Linking.getInitialURL();

// Escuchar cambios de URL
Linking.addEventListener('url', (event) => {
  console.log(event.url);
});

// Abrir URL
await Linking.openURL('https://example.com');
```

### 19. **Geolocalización - Expo Location**

```typescript
import * as Location from 'expo-location';

const { status } = await Location.requestForegroundPermissionsAsync();
if (status !== 'granted') {
  return;
}

const location = await Location.getCurrentPositionAsync({});
console.log(location);
```

### 20. **Compartir - Expo Sharing**

```typescript
import * as Sharing from 'expo-sharing';

const shareImage = async (uri: string) => {
  const isAvailable = await Sharing.isAvailableAsync();
  if (isAvailable) {
    await Sharing.shareAsync(uri);
  }
};
```

---

## 🔧 Configuraciones Adicionales

### Babel Config (ya configurado)
```javascript
// babel.config.js
module.exports = function(api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: ['react-native-reanimated/plugin'], // Debe ser el último
  };
};
```

### TypeScript Paths (ya configurado)
```json
// tsconfig.json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  }
}
```

---

## 📚 Recursos Adicionales

- [Expo Documentation](https://docs.expo.dev/)
- [React Navigation](https://reactnavigation.org/)
- [React Query](https://tanstack.com/query/latest)
- [React Hook Form](https://react-hook-form.com/)
- [Redux Toolkit](https://redux-toolkit.js.org/)

---

**Nota**: Esta es una guía rápida. Consulta la documentación oficial de cada librería para más detalles.

