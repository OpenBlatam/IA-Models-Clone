'use client';

import React, { useState } from 'react';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { useAuth } from '@/lib/contexts/AuthContext';
import { toastMessages, showError } from '@/lib/utils/toastUtils';
import { useMutation } from '@/lib/hooks/useMutation';

interface RegisterModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSwitchToLogin: () => void;
}

export const RegisterModal: React.FC<RegisterModalProps> = ({
  isOpen,
  onClose,
  onSwitchToLogin,
}) => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const { register } = useAuth();

  const registerMutation = useMutation<boolean, { name: string; email: string; password: string }>({
    mutationFn: async ({ name, email, password }) => {
      if (password.length < 6) {
        throw new Error('Password must be at least 6 characters');
      }
      const success = await register(email, password, name);
      if (!success) {
        throw new Error('Registration failed. Please try again.');
      }
      return success;
    },
    onSuccess: () => {
      onClose();
      setName('');
      setEmail('');
      setPassword('');
      setConfirmPassword('');
      onSwitchToLogin();
    },
    errorMessage: 'Registration failed. Please try again.',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (password !== confirmPassword) {
      showError('Passwords do not match');
      return;
    }

    registerMutation.mutate({ name, email, password });
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Sign up" size="sm">
      <form onSubmit={handleSubmit} className="space-y-4">
        <Input
          label="Name"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
          placeholder="Enter your full name"
          fullWidth
        />

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
          minLength={6}
          placeholder="Enter your password"
          helperText="Must be at least 6 characters"
          fullWidth
        />

        <Input
          label="Confirm Password"
          type="password"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          required
          minLength={6}
          placeholder="Confirm your password"
          fullWidth
        />

        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={onSwitchToLogin}
            className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300"
          >
            Already have an account? Sign in
          </button>
        </div>

        <div className="flex space-x-3 pt-4">
          <Button
            type="submit"
            isLoading={registerMutation.isLoading}
            className="flex-1"
          >
            Sign up
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

