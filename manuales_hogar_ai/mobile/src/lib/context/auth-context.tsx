/**
 * Auth Context
 * ============
 * Authentication context with Google OAuth
 */

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import * as AuthSession from 'expo-auth-session';
import * as WebBrowser from 'expo-web-browser';
import * as SecureStore from 'expo-secure-store';
import { useRouter } from 'expo-router';
import { useTranslation } from 'react-i18next';
import { Alert } from 'react-native';
import { apiClient } from '@/services/api/api-client';
import type { User } from '@/types/auth';

WebBrowser.maybeCompleteAuthSession();

interface AuthContextValue {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  signInWithGoogle: () => Promise<void>;
  signOut: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();
  const { t } = useTranslation();

  // Configure Google OAuth
  const discovery = {
    authorizationEndpoint: 'https://accounts.google.com/o/oauth2/v2/auth',
    tokenEndpoint: 'https://oauth2.googleapis.com/token',
    revocationEndpoint: 'https://oauth2.googleapis.com/revoke',
  };

  const [request, response, promptAsync] = AuthSession.useAuthRequest(
    {
      clientId: process.env.EXPO_PUBLIC_GOOGLE_CLIENT_ID || '',
      scopes: ['openid', 'profile', 'email'],
      redirectUri: AuthSession.makeRedirectUri({
        scheme: 'manuales-hogar-ai',
        path: 'auth',
      }),
      responseType: AuthSession.ResponseType.Token,
    },
    discovery
  );

  useEffect(() => {
    checkAuthState();
  }, []);

  useEffect(() => {
    if (response?.type === 'success') {
      handleAuthResponse(response);
    } else if (response?.type === 'error') {
      Alert.alert(t('errors.error'), response.error?.message || t('errors.unknownError'));
    }
  }, [response]);

  async function checkAuthState() {
    try {
      const token = await SecureStore.getItemAsync('auth_token');
      const userData = await SecureStore.getItemAsync('user_data');

      if (token && userData) {
        const parsedUser = JSON.parse(userData);
        setUser(parsedUser);
        // Verify token is still valid
        await refreshUser();
      }
    } catch (error) {
      console.error('Error checking auth state:', error);
    } finally {
      setIsLoading(false);
    }
  }

  async function handleAuthResponse(response: AuthSession.AuthRequestResponse) {
    try {
      if (response.type === 'success' && response.authentication?.accessToken) {
        const accessToken = response.authentication.accessToken;

        // Exchange token with backend
        const backendResponse = await apiClient.post<{ user: User; token: string }>(
          '/api/v1/auth/google',
          {
            access_token: accessToken,
          }
        );

        // Store tokens
        await SecureStore.setItemAsync('auth_token', backendResponse.token);
        await SecureStore.setItemAsync('user_data', JSON.stringify(backendResponse.user));

        setUser(backendResponse.user);
      }
    } catch (error) {
      console.error('Error handling auth response:', error);
      Alert.alert(t('errors.error'), t('errors.unknownError'));
    }
  }

  async function signInWithGoogle() {
    try {
      if (!request) {
        Alert.alert(t('errors.error'), 'OAuth not configured');
        return;
      }

      await promptAsync();
    } catch (error) {
      console.error('Error signing in with Google:', error);
      Alert.alert(t('errors.error'), t('errors.unknownError'));
    }
  }

  async function signOut() {
    try {
      Alert.alert(
        t('auth.signOut'),
        t('auth.signOutConfirm'),
        [
          { text: t('common.cancel'), style: 'cancel' },
          {
            text: t('common.confirm'),
            style: 'destructive',
            onPress: async () => {
              await SecureStore.deleteItemAsync('auth_token');
              await SecureStore.deleteItemAsync('user_data');
              setUser(null);
              router.replace('/(auth)/login');
            },
          },
        ]
      );
    } catch (error) {
      console.error('Error signing out:', error);
    }
  }

  async function refreshUser() {
    try {
      const userResponse = await apiClient.get<User>('/api/v1/auth/me');
      setUser(userResponse);
      await SecureStore.setItemAsync('user_data', JSON.stringify(userResponse));
    } catch (error) {
      // Token might be invalid, sign out
      await SecureStore.deleteItemAsync('auth_token');
      await SecureStore.deleteItemAsync('user_data');
      setUser(null);
    }
  }

  const value: AuthContextValue = {
    user,
    isLoading,
    isAuthenticated: !!user,
    signInWithGoogle,
    signOut,
    refreshUser,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}




