import { AuthRepository } from '../data/auth-repository';
import * as SecureStore from 'react-native-encrypted-storage';
import { STORAGE_KEYS } from '@/constants/config';
import type { LoginCredentials, RegisterData } from './auth-types';
import type { User, Session } from '@/types';

export class AuthService {
  constructor(private repository: AuthRepository = new AuthRepository()) {}

  async login(credentials: LoginCredentials): Promise<Session> {
    const session = await this.repository.login(credentials);
    
    // Store session
    await SecureStore.setItem(STORAGE_KEYS.SESSION_ID, session.session_id);
    await SecureStore.setItem(STORAGE_KEYS.USER_ID, session.user_id);
    await SecureStore.setItem(STORAGE_KEYS.USER_DATA, JSON.stringify(session.user));
    
    return session;
  }

  async register(data: RegisterData): Promise<User> {
    return this.repository.register(data);
  }

  async logout(): Promise<void> {
    const sessionId = await SecureStore.getItem(STORAGE_KEYS.SESSION_ID);
    if (sessionId) {
      await this.repository.logout(sessionId);
    }
    
    // Clear storage
    await SecureStore.removeItem(STORAGE_KEYS.SESSION_ID);
    await SecureStore.removeItem(STORAGE_KEYS.USER_ID);
    await SecureStore.removeItem(STORAGE_KEYS.USER_DATA);
  }

  async verifySession(): Promise<User | null> {
    const sessionId = await SecureStore.getItem(STORAGE_KEYS.SESSION_ID);
    if (!sessionId) {
      return null;
    }

    try {
      return await this.repository.verifySession(sessionId);
    } catch {
      // Session invalid, clear storage
      await this.logout();
      return null;
    }
  }
}

