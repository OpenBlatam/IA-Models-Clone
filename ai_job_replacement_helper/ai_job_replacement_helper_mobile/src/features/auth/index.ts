// Domain
export { AuthService } from './domain/auth-service';
export type { LoginCredentials, RegisterData, AuthState } from './domain/auth-types';

// Data
export { AuthRepository } from './data/auth-repository';

// Presentation
export { useAuthStore } from './presentation/auth-store';

// Validators
export { loginSchema, registerSchema, type LoginFormData, type RegisterFormData } from '@/modules/auth/validators';

// Constants
export { AUTH_STORAGE_KEYS, AUTH_ERRORS } from '@/modules/auth/constants';

