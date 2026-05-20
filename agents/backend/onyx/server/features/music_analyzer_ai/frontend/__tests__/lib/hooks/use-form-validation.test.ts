import { renderHook, act } from '@testing-library/react';
import { useFormValidation } from '@/lib/hooks/use-form-validation';
import { z } from 'zod';

describe('useFormValidation', () => {
  const schema = z.object({
    email: z.string().email('Invalid email'),
    password: z.string().min(8, 'Password must be at least 8 characters'),
    age: z.number().min(18, 'Must be at least 18'),
  });

  it('should initialize with initial values', () => {
    const initialValues = { email: 'test@example.com', password: '' };
    const { result } = renderHook(() =>
      useFormValidation({
        schema: schema.partial(),
        initialValues,
      })
    );

    expect(result.current.values.email).toBe('test@example.com');
    expect(result.current.values.password).toBe('');
    expect(result.current.errors).toEqual({});
    expect(result.current.isValid).toBe(true);
  });

  it('should validate field on blur when validateOnBlur is true', async () => {
    const { result } = renderHook(() =>
      useFormValidation({
        schema,
        validateOnBlur: true,
      })
    );

    act(() => {
      result.current.setValue('email', 'invalid-email');
    });

    act(() => {
      result.current.handleBlur('email')();
    });

    expect(result.current.touched.email).toBe(true);
    expect(result.current.errors.email).toBeDefined();
  });

  it('should validate field on change when validateOnChange is true', async () => {
    const { result } = renderHook(() =>
      useFormValidation({
        schema,
        validateOnChange: true,
      })
    );

    act(() => {
      result.current.setValue('email', 'invalid-email');
    });

    // Wait for validation
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(result.current.errors.email).toBeDefined();
  });

  it('should validate entire form', () => {
    const { result } = renderHook(() =>
      useFormValidation({
        schema,
      })
    );

    act(() => {
      result.current.setValue('email', 'test@example.com');
      result.current.setValue('password', 'password123');
      result.current.setValue('age', 25);
    });

    const validationResult = result.current.validate();

    expect(validationResult.success).toBe(true);
    expect(validationResult.data).toEqual({
      email: 'test@example.com',
      password: 'password123',
      age: 25,
    });
    expect(result.current.errors).toEqual({});
  });

  it('should return errors when validation fails', () => {
    const { result } = renderHook(() =>
      useFormValidation({
        schema,
      })
    );

    act(() => {
      result.current.setValue('email', 'invalid-email');
      result.current.setValue('password', 'short');
      result.current.setValue('age', 15);
    });

    const validationResult = result.current.validate();

    expect(validationResult.success).toBe(false);
    expect(validationResult.errors).toBeDefined();
    expect(Object.keys(validationResult.errors || {})).toContain('email');
    expect(Object.keys(validationResult.errors || {})).toContain('password');
    expect(Object.keys(validationResult.errors || {})).toContain('age');
  });

  it('should set field value', () => {
    const { result } = renderHook(() =>
      useFormValidation({
        schema: schema.partial(),
      })
    );

    act(() => {
      result.current.setValue('email', 'new@example.com');
    });

    expect(result.current.values.email).toBe('new@example.com');
  });

  it('should set multiple values', () => {
    const { result } = renderHook(() =>
      useFormValidation({
        schema: schema.partial(),
      })
    );

    act(() => {
      result.current.setValues({
        email: 'test@example.com',
        password: 'password123',
      });
    });

    expect(result.current.values.email).toBe('test@example.com');
    expect(result.current.values.password).toBe('password123');
  });

  it('should set field error manually', () => {
    const { result } = renderHook(() =>
      useFormValidation({
        schema: schema.partial(),
      })
    );

    act(() => {
      result.current.setError('email', 'Custom error message');
    });

    expect(result.current.errors.email).toEqual(['Custom error message']);
  });

  it('should clear field error', () => {
    const { result } = renderHook(() =>
      useFormValidation({
        schema: schema.partial(),
      })
    );

    act(() => {
      result.current.setError('email', 'Error message');
    });

    expect(result.current.errors.email).toBeDefined();

    act(() => {
      result.current.clearError('email');
    });

    expect(result.current.errors.email).toBeUndefined();
  });

  it('should clear all errors', () => {
    const { result } = renderHook(() =>
      useFormValidation({
        schema: schema.partial(),
      })
    );

    act(() => {
      result.current.setError('email', 'Error 1');
      result.current.setError('password', 'Error 2');
    });

    expect(Object.keys(result.current.errors).length).toBeGreaterThan(0);

    act(() => {
      result.current.clearAllErrors();
    });

    expect(Object.keys(result.current.errors).length).toBe(0);
  });

  it('should handle form submission with valid data', async () => {
    const onSubmit = jest.fn().mockResolvedValue(undefined);

    const { result } = renderHook(() =>
      useFormValidation({
        schema,
        onSubmit,
      })
    );

    act(() => {
      result.current.setValue('email', 'test@example.com');
      result.current.setValue('password', 'password123');
      result.current.setValue('age', 25);
    });

    await act(async () => {
      await result.current.handleSubmit();
    });

    expect(onSubmit).toHaveBeenCalledWith({
      email: 'test@example.com',
      password: 'password123',
      age: 25,
    });
    expect(result.current.isSubmitting).toBe(false);
  });

  it('should not submit when validation fails', async () => {
    const onSubmit = jest.fn();

    const { result } = renderHook(() =>
      useFormValidation({
        schema,
        onSubmit,
      })
    );

    act(() => {
      result.current.setValue('email', 'invalid-email');
    });

    await act(async () => {
      await result.current.handleSubmit();
    });

    expect(onSubmit).not.toHaveBeenCalled();
    expect(result.current.touched.email).toBe(true);
  });

  it('should handle submission errors', async () => {
    const onSubmit = jest.fn().mockRejectedValue(new Error('Submission failed'));

    const { result } = renderHook(() =>
      useFormValidation({
        schema,
        onSubmit,
      })
    );

    act(() => {
      result.current.setValue('email', 'test@example.com');
      result.current.setValue('password', 'password123');
      result.current.setValue('age', 25);
    });

    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();

    await act(async () => {
      await result.current.handleSubmit();
    });

    expect(result.current.isSubmitting).toBe(false);
    expect(consoleSpy).toHaveBeenCalled();

    consoleSpy.mockRestore();
  });

  it('should reset form to initial values', () => {
    const initialValues = { email: 'initial@example.com' };
    const { result } = renderHook(() =>
      useFormValidation({
        schema: schema.partial(),
        initialValues,
      })
    );

    act(() => {
      result.current.setValue('email', 'changed@example.com');
      result.current.setValue('password', 'password123');
      result.current.setError('email', 'Error');
    });

    act(() => {
      result.current.reset();
    });

    expect(result.current.values.email).toBe('initial@example.com');
    expect(result.current.values.password).toBeUndefined();
    expect(result.current.errors).toEqual({});
    expect(result.current.touched).toEqual({});
    expect(result.current.isSubmitting).toBe(false);
  });

  it('should handle change event', () => {
    const { result } = renderHook(() =>
      useFormValidation({
        schema: schema.partial(),
      })
    );

    const handleChange = result.current.handleChange('email');

    act(() => {
      handleChange('new@example.com');
    });

    expect(result.current.values.email).toBe('new@example.com');
  });

  it('should prevent default on form submit', async () => {
    const { result } = renderHook(() =>
      useFormValidation({
        schema,
      })
    );

    const mockEvent = {
      preventDefault: jest.fn(),
    } as unknown as React.FormEvent;

    act(() => {
      result.current.setValue('email', 'test@example.com');
      result.current.setValue('password', 'password123');
      result.current.setValue('age', 25);
    });

    await act(async () => {
      await result.current.handleSubmit(mockEvent);
    });

    expect(mockEvent.preventDefault).toHaveBeenCalled();
  });

  it('should calculate isValid correctly', () => {
    const { result } = renderHook(() =>
      useFormValidation({
        schema: schema.partial(),
      })
    );

    expect(result.current.isValid).toBe(true);

    act(() => {
      result.current.setError('email', 'Error');
    });

    expect(result.current.isValid).toBe(false);

    act(() => {
      result.current.clearError('email');
    });

    expect(result.current.isValid).toBe(true);
  });
});

