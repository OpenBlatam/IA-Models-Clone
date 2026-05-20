'use client';

import React, { useState } from 'react';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { useAuth } from '@/lib/contexts/AuthContext';
import { toastMessages, showError } from '@/lib/utils/toastUtils';
import { useMutation } from '@/lib/hooks/useMutation';

interface LoginModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSwitchToRegister: () => void;
}

export const LoginModal: React.FC<LoginModalProps> = ({
  isOpen,
  onClose,
  onSwitchToRegister,
}) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { login } = useAuth();

  const loginMutation = useMutation<boolean, { email: string; password: string }>({
    mutationFn: async ({ email, password }) => {
      const success = await login(email, password);
      if (!success) {
        throw new Error('Login failed. Please check your credentials.');
      }
      return success;
    },
    onSuccess: () => {
      onClose();
      setEmail('');
      setPassword('');
    },
    errorMessage: 'Login failed. Please check your credentials.',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    loginMutation.mutate({ email, password });
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Sign in" size="sm">
      <form onSubmit={handleSubmit} className="space-y-4">
        <Input
          label="Email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          placeholder="Enter your email"
          fullWidth
        />

        <Input
          label="Password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          placeholder="Enter your password"
          fullWidth
        />

        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={onSwitchToRegister}
            className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300"
          >
            Don't have an account? Sign up
          </button>
        </div>

        <div className="flex space-x-3 pt-4">
          <Button
            type="submit"
            isLoading={loginMutation.isLoading}
            className="flex-1"
          >
            Sign in
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={onClose}
            className="flex-1"
          >
            Cancel
          </Button>
        </div>
      </form>
    </Modal>
  );
};

