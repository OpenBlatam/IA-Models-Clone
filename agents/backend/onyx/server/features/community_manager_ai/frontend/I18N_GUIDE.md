# Guía de Internacionalización (i18n)

## 🌍 Idiomas Soportados

- **Español (es)** - Idioma por defecto
- **English (en)**
- **Français (fr)**
- **Português (pt)**

## 📁 Estructura

```
i18n/
├── config.ts          # Configuración de idiomas
├── request.ts         # Configuración de next-intl
├── routing.ts         # Routing con i18n
└── messages/
    ├── es.json        # Traducciones en español
    ├── en.json        # Traducciones en inglés
    ├── fr.json        # Traducciones en francés
    └── pt.json        # Traducciones en portugués
```

## 🚀 Uso en Componentes

### Hook useTranslations

```typescript
'use client';

import { useTranslations } from 'next-intl';

export const MyComponent = () => {
  const t = useTranslations('posts');
  const tCommon = useTranslations('common');

  return (
    <div>
      <h1>{t('title')}</h1>
      <button>{tCommon('save')}</button>
    </div>
  );
};
```

### Server Components

```typescript
import { useTranslations } from 'next-intl';

export default async function Page() {
  const t = await useTranslations('dashboard');

  return <h1>{t('title')}</h1>;
}
```

### Navegación con i18n

```typescript
import { Link, usePathname, useRouter } from '@/i18n/routing';

// Link con locale automático
<Link href="/dashboard">Dashboard</Link>

// Navegación programática
const router = useRouter();
router.push('/posts');

// Pathname con locale
const pathname = usePathname();
```

## 🎨 Selector de Idioma

El componente `LanguageSelector` está integrado en el Header:

```typescript
import { LanguageSelector } from '@/components/ui/LanguageSelector';

<LanguageSelector />
```

## 📝 Agregar Nuevas Traducciones

1. Agregar la clave en todos los archivos JSON:

```json
// es.json
{
  "newSection": {
    "title": "Nuevo Título"
  }
}

// en.json
{
  "newSection": {
    "title": "New Title"
  }
}
```

2. Usar en componentes:

```typescript
const t = useTranslations('newSection');
<h1>{t('title')}</h1>
```

## 🔧 Configuración

### Agregar Nuevo Idioma

1. Agregar en `i18n/config.ts`:

```typescript
export const locales = ['es', 'en', 'fr', 'pt', 'de'] as const;
```

2. Crear archivo de traducciones: `i18n/messages/de.json`
3. Agregar nombre y bandera en `localeNames` y `localeFlags`

### Cambiar Idioma por Defecto

En `i18n/config.ts`:

```typescript
export const defaultLocale: Locale = 'en';
```

## 🌐 Variables en Traducciones

```json
{
  "platforms": {
    "disconnectConfirm": "¿Estás seguro de desconectar {platform}?"
  }
}
```

```typescript
t('platforms.disconnectConfirm', { platform: 'Facebook' })
```

## 📱 Rutas con Locale

Las rutas incluyen el locale automáticamente:
- `/es/dashboard`
- `/en/dashboard`
- `/fr/dashboard`
- `/pt/dashboard`

El middleware redirige automáticamente a la ruta con locale.

## ✅ Mejores Prácticas

1. **Usar namespaces**: Organizar traducciones por sección
2. **Claves descriptivas**: `posts.title` en lugar de `t1`
3. **Reutilizar comunes**: Usar `common` para textos compartidos
4. **Validar traducciones**: Asegurar que todas las claves existan en todos los idiomas
5. **Contexto**: Agregar comentarios en JSON si es necesario

## 🐛 Troubleshooting

### Error: "useTranslations must be called from within a NextIntlClientProvider"

Asegúrate de que el componente esté dentro del layout con `[locale]`.

### Rutas no funcionan

Verifica que el middleware esté configurado correctamente y que las páginas estén en `app/[locale]/`.

### Traducciones no aparecen

1. Verifica que el archivo JSON existe
2. Verifica que la clave existe en el JSON
3. Verifica que el namespace es correcto



