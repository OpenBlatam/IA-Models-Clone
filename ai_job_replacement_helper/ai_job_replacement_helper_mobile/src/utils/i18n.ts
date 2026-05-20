import * as Localization from 'expo-localization';
import { I18n } from 'i18n-js';

// Translation files (to be expanded)
const translations = {
  en: {
    auth: {
      login: 'Sign In',
      register: 'Sign Up',
      email: 'Email',
      password: 'Password',
      confirmPassword: 'Confirm Password',
      username: 'Username',
      forgotPassword: 'Forgot Password?',
      noAccount: "Don't have an account?",
      hasAccount: 'Already have an account?',
    },
    common: {
      loading: 'Loading...',
      error: 'An error occurred',
      retry: 'Try Again',
      cancel: 'Cancel',
      confirm: 'Confirm',
      save: 'Save',
      delete: 'Delete',
      edit: 'Edit',
      close: 'Close',
    },
    dashboard: {
      welcome: 'Welcome back!',
      yourProgress: 'Your Progress',
      statistics: 'Statistics',
      quickActions: 'Quick Actions',
      recommendedSteps: 'Recommended Next Steps',
    },
    jobs: {
      findJob: 'Find Your Dream Job',
      noJobs: 'No more jobs available',
      refresh: 'Refresh',
      apply: 'Apply',
      save: 'Save',
      like: 'Like',
      dislike: 'Dislike',
    },
  },
  es: {
    auth: {
      login: 'Iniciar Sesión',
      register: 'Registrarse',
      email: 'Correo Electrónico',
      password: 'Contraseña',
      confirmPassword: 'Confirmar Contraseña',
      username: 'Nombre de Usuario',
      forgotPassword: '¿Olvidaste tu contraseña?',
      noAccount: '¿No tienes una cuenta?',
      hasAccount: '¿Ya tienes una cuenta?',
    },
    common: {
      loading: 'Cargando...',
      error: 'Ocurrió un error',
      retry: 'Intentar de Nuevo',
      cancel: 'Cancelar',
      confirm: 'Confirmar',
      save: 'Guardar',
      delete: 'Eliminar',
      edit: 'Editar',
      close: 'Cerrar',
    },
    dashboard: {
      welcome: '¡Bienvenido de nuevo!',
      yourProgress: 'Tu Progreso',
      statistics: 'Estadísticas',
      quickActions: 'Acciones Rápidas',
      recommendedSteps: 'Pasos Recomendados',
    },
    jobs: {
      findJob: 'Encuentra tu Trabajo Ideal',
      noJobs: 'No hay más trabajos disponibles',
      refresh: 'Actualizar',
      apply: 'Aplicar',
      save: 'Guardar',
      like: 'Me Gusta',
      dislike: 'No Me Gusta',
    },
  },
};

const i18n = new I18n(translations);

// Set the locale once at the beginning of your app
i18n.locale = Localization.locale;

// When a value is missing from a language it'll fallback to another language with the key
i18n.enableFallback = true;

// Set default locale
i18n.defaultLocale = 'en';

export default i18n;


