# Utilities Documentation

Colección completa de utilidades para la aplicación móvil.

## 📚 Categorías

### 1. Strings (`strings.ts`)

Utilidades para manipulación de strings.

```tsx
import { truncate, capitalize, slugify, isValidEmail } from '@/utils/strings';

// Truncar texto
truncate('Long text here', 10); // 'Long te...'

// Capitalizar
capitalize('hello world'); // 'Hello world'

// Slugify
slugify('Hello World!'); // 'hello-world'

// Validar email
isValidEmail('user@example.com'); // true
```

**Funciones principales:**
- `truncate()` - Truncar texto con sufijo
- `capitalize()` - Capitalizar primera letra
- `capitalizeWords()` - Capitalizar cada palabra
- `camelCase()` - Convertir a camelCase
- `kebabCase()` - Convertir a kebab-case
- `snakeCase()` - Convertir a snake_case
- `slugify()` - Crear slug URL-friendly
- `removeAccents()` - Remover acentos
- `stripHtml()` - Remover HTML tags
- `escapeHtml()` / `unescapeHtml()` - Escapar/desescapar HTML
- `extractEmails()` - Extraer emails de texto
- `extractUrls()` - Extraer URLs de texto
- `extractHashtags()` - Extraer hashtags
- `extractMentions()` - Extraer menciones
- `maskEmail()` - Enmascarar email
- `maskPhone()` - Enmascarar teléfono
- `pluralize()` - Pluralizar palabras
- `formatInitials()` - Formatear iniciales
- `isValidEmail()` - Validar email
- `isValidUrl()` - Validar URL
- `generateId()` - Generar ID aleatorio
- `generateUUID()` - Generar UUID

### 2. Arrays (`arrays.ts`)

Utilidades para manipulación de arrays.

```tsx
import { chunk, unique, groupBy, sortBy } from '@/utils/arrays';

// Dividir en chunks
chunk([1, 2, 3, 4, 5], 2); // [[1, 2], [3, 4], [5]]

// Valores únicos
unique([1, 2, 2, 3]); // [1, 2, 3]

// Agrupar por propiedad
groupBy(users, 'role');

// Ordenar
sortBy(users, 'name', 'asc');
```

**Funciones principales:**
- `chunk()` - Dividir en chunks
- `unique()` - Valores únicos
- `uniqueBy()` - Únicos por propiedad
- `groupBy()` - Agrupar por clave
- `sortBy()` - Ordenar por propiedad
- `shuffle()` - Mezclar aleatoriamente
- `sample()` - Muestra aleatoria
- `flatten()` - Aplanar arrays anidados
- `difference()` - Diferencia entre arrays
- `intersection()` - Intersección
- `union()` - Unión
- `partition()` - Particionar por condición
- `zip()` / `unzip()` - Combinar/separar arrays
- `range()` - Generar rango de números
- `compact()` - Remover valores falsy
- `take()` / `takeRight()` - Tomar elementos
- `drop()` / `dropRight()` - Remover elementos
- `findIndex()` / `findLastIndex()` - Encontrar índice

### 3. Objects (`objects.ts`)

Utilidades para manipulación de objetos.

```tsx
import { pick, omit, deepClone, merge } from '@/utils/objects';

// Seleccionar propiedades
pick(user, ['name', 'email']);

// Omitir propiedades
omit(user, ['password']);

// Clonar profundamente
const cloned = deepClone(original);

// Fusionar objetos
merge(target, source1, source2);
```

**Funciones principales:**
- `pick()` - Seleccionar propiedades
- `omit()` - Omitir propiedades
- `deepClone()` - Clonar profundamente
- `merge()` - Fusionar objetos
- `isEmpty()` - Verificar si está vacío
- `isEqual()` - Comparar objetos
- `get()` - Obtener valor por path
- `set()` - Establecer valor por path
- `flattenObject()` - Aplanar objeto
- `invert()` - Invertir clave-valor
- `mapKeys()` - Mapear claves
- `mapValues()` - Mapear valores
- `defaults()` - Valores por defecto
- `has()` - Verificar existencia de path

### 4. Numbers (`numbers.ts`)

Utilidades para manipulación de números.

```tsx
import { clamp, formatNumber, formatCurrency } from '@/utils/numbers';

// Limitar valor
clamp(150, 0, 100); // 100

// Formatear número
formatNumber(1234.56, { decimals: 2 }); // '1,234.56'

// Formatear moneda
formatCurrency(1234.56, 'USD'); // '$1,234.56'
```

**Funciones principales:**
- `clamp()` - Limitar entre min y max
- `random()` - Número aleatorio entero
- `randomFloat()` - Número aleatorio flotante
- `round()` / `floor()` / `ceil()` - Redondear
- `padStart()` / `padEnd()` - Rellenar con ceros
- `formatNumber()` - Formatear con separadores
- `formatCurrency()` - Formatear como moneda
- `formatPercentage()` - Formatear porcentaje
- `parseNumber()` - Parsear string a número
- `isEven()` / `isOdd()` - Par/impar
- `isInteger()` / `isFloat()` - Tipo de número
- `isPositive()` / `isNegative()` - Signo
- `isBetween()` - Verificar rango
- `sum()` - Sumar array
- `average()` - Promedio
- `min()` / `max()` - Mínimo/máximo
- `median()` - Mediana
- `mode()` - Moda
- `variance()` - Varianza
- `standardDeviation()` - Desviación estándar
- `lerp()` - Interpolación lineal
- `normalize()` / `denormalize()` - Normalizar
- `toRadians()` / `toDegrees()` - Conversión angular

### 5. Dates (`dates.ts`)

Utilidades para manipulación de fechas.

```tsx
import { formatDate, addDays, isToday, differenceInDays } from '@/utils/dates';

// Formatear fecha
formatDate(new Date(), 'relative'); // '2 hours ago'

// Agregar días
addDays(new Date(), 7);

// Verificar si es hoy
isToday(date);

// Diferencia en días
differenceInDays(date1, date2);
```

**Funciones principales:**
- `formatDate()` - Formatear con múltiples formatos
- `addDays()` / `addHours()` / `addMinutes()` - Agregar tiempo
- `addMonths()` / `addYears()` - Agregar períodos
- `startOfDay()` / `endOfDay()` - Inicio/fin del día
- `startOfWeek()` / `endOfWeek()` - Inicio/fin de semana
- `startOfMonth()` / `endOfMonth()` - Inicio/fin de mes
- `isToday()` / `isYesterday()` / `isTomorrow()` - Verificaciones
- `isSameDay()` / `isSameMonth()` / `isSameYear()` - Comparaciones
- `differenceInDays()` / `differenceInHours()` - Diferencias
- `isPast()` / `isFuture()` - Verificar tiempo
- `isValidDate()` - Validar fecha
- `parseDate()` - Parsear a Date
- `getDaysInMonth()` - Días del mes
- `getWeekNumber()` - Número de semana

### 6. URLs (`urls.ts`)

Utilidades para URLs y deep linking.

```tsx
import { buildUrl, parseUrl, openUrl, getDeepLink } from '@/utils/urls';

// Construir URL con parámetros
buildUrl('https://api.com', { page: 1, limit: 10 });

// Parsear URL
parseUrl('https://api.com?page=1');

// Abrir URL
await openUrl('https://example.com');

// Crear deep link
getDeepLink('/video/123', { autoplay: true });
```

**Funciones principales:**
- `buildUrl()` - Construir URL con parámetros
- `parseUrl()` - Parsear URL
- `getQueryParams()` - Obtener query params
- `addQueryParams()` - Agregar query params
- `removeQueryParams()` - Remover query params
- `canOpenUrl()` - Verificar si se puede abrir
- `openUrl()` - Abrir URL
- `getDeepLink()` - Crear deep link
- `parseDeepLink()` - Parsear deep link
- `isValidUrl()` - Validar URL
- `isAbsoluteUrl()` - Verificar si es absoluta
- `normalizeUrl()` - Normalizar URL
- `getDomain()` - Obtener dominio
- `getProtocol()` - Obtener protocolo
- `getPath()` - Obtener path
- `encodeUrl()` / `decodeUrl()` - Codificar/decodificar
- `sanitizeUrl()` - Sanitizar URL

### 7. Permissions (`permissions.ts`)

Utilidades para manejo de permisos.

```tsx
import {
  requestCameraPermission,
  requestMediaLibraryPermission,
  openSettings,
} from '@/utils/permissions';

// Solicitar permiso
const { granted } = await requestCameraPermission();

// Abrir configuración
openSettings();
```

**Funciones principales:**
- `requestMediaLibraryPermission()` - Permiso de galería
- `requestCameraPermission()` - Permiso de cámara
- `requestImagePickerPermission()` - Permiso de image picker
- `requestNotificationPermission()` - Permiso de notificaciones
- `checkMediaLibraryPermission()` - Verificar permiso
- `checkCameraPermission()` - Verificar permiso
- `checkNotificationPermission()` - Verificar permiso
- `openSettings()` - Abrir configuración
- `showPermissionAlert()` - Mostrar alerta
- `requestMultiplePermissions()` - Múltiples permisos

### 8. Images (`images.ts`)

Utilidades para manejo de imágenes.

```tsx
import {
  pickImage,
  takePhoto,
  getImageInfo,
  calculateDimensions,
} from '@/utils/images';

// Seleccionar imagen
const result = await pickImage();

// Tomar foto
const photo = await takePhoto();

// Obtener información
const info = await getImageInfo(uri);
```

**Funciones principales:**
- `getImageInfo()` - Obtener información de imagen
- `pickImage()` - Seleccionar de galería
- `pickMultipleImages()` - Seleccionar múltiples
- `takePhoto()` - Tomar foto
- `calculateAspectRatio()` - Calcular aspect ratio
- `calculateDimensions()` - Calcular dimensiones
- `getImageType()` - Obtener tipo MIME
- `isValidImageFormat()` - Validar formato
- `formatImageSize()` - Formatear tamaño
- `getImageDimensions()` - Obtener dimensiones
- `isImageLandscape()` / `isImagePortrait()` - Orientación
- `isImageSquare()` - Verificar si es cuadrado

### 9. Performance (`performance.ts`)

Utilidades para optimización de performance.

```tsx
import { debounce, throttle, memoize, measurePerformance } from '@/utils/performance';

// Debounce
const debouncedSearch = debounce(searchFunction, 300);

// Throttle
const throttledScroll = throttle(handleScroll, 100);

// Memoizar
const memoizedCalc = memoize(expensiveCalculation);

// Medir performance
measurePerformance('MyFunction', () => {
  // código
});
```

**Funciones principales:**
- `debounce()` - Debounce de función
- `throttle()` - Throttle de función
- `memoize()` - Memoización
- `measurePerformance()` - Medir performance síncrona
- `measureAsyncPerformance()` - Medir performance asíncrona
- `createLazyLoader()` - Lazy loading
- `batchUpdates()` - Agrupar actualizaciones
- `requestIdleCallback()` - Callback en idle
- `cancelIdleCallback()` - Cancelar callback

### 10. Security (`security.ts`)

Utilidades de seguridad.

```tsx
import {
  sanitizeInput,
  isValidPassword,
  maskSensitiveData,
  detectXSS,
} from '@/utils/security';

// Sanitizar input
const safe = sanitizeInput(userInput);

// Validar contraseña
const { valid, errors } = isValidPassword(password);

// Enmascarar datos
maskSensitiveData('1234567890', 4); // '1234****7890'

// Detectar XSS
if (detectXSS(input)) {
  // peligroso
}
```

**Funciones principales:**
- `sanitizeInput()` - Sanitizar input de usuario
- `sanitizeHtml()` - Sanitizar HTML
- `escapeRegex()` - Escapar regex
- `generateSecureToken()` - Generar token seguro
- `hashString()` - Hash de string
- `maskSensitiveData()` - Enmascarar datos sensibles
- `isValidPassword()` - Validar contraseña
- `validateUrl()` - Validar URL
- `sanitizeUrl()` - Sanitizar URL
- `detectXSS()` - Detectar XSS
- `detectSQLInjection()` - Detectar SQL injection
- `sanitizeFilename()` - Sanitizar nombre de archivo

### 11. Format (`format.ts`)

Utilidades de formateo (ya existente, mejorado).

### 12. Storage (`storage.ts`)

Utilidades de almacenamiento (ya existente).

## 🎯 Uso Recomendado

### Importar desde el índice central

```tsx
import {
  truncate,
  formatDate,
  clamp,
  debounce,
} from '@/utils';
```

### Importar desde módulos específicos

```tsx
import { truncate } from '@/utils/strings';
import { formatDate } from '@/utils/dates';
```

## 📝 Notas

- Todas las funciones son puras cuando es posible
- Funciones asíncronas devuelven Promises
- Validaciones incluidas en funciones críticas
- TypeScript estricto en todas las funciones
- Documentación JSDoc en funciones complejas

## 🔒 Seguridad

Las utilidades de seguridad deben usarse siempre que:
- Se reciba input del usuario
- Se muestre contenido dinámico
- Se manejen URLs o paths
- Se procesen archivos

## ⚡ Performance

Las utilidades de performance ayudan a:
- Optimizar renders
- Reducir llamadas a API
- Mejorar UX con debounce/throttle
- Medir y optimizar código lento

---

**Total de utilidades**: 200+ funciones  
**Categorías**: 12 módulos  
**Type Safety**: 100% TypeScript


