# React Native Security Pattern: Input Sanitization & XSS Prevention

## Overview
Comprehensive security patterns for React Native applications focusing on input sanitization, XSS prevention, and secure data handling.

## Core Security Principles

### 1. Input Sanitization
```typescript
// Input sanitization utilities
const sanitizeInput = (input: string): string => {
  return input
    .replace(/[<>]/g, '') // Remove angle brackets
    .replace(/javascript:/gi, '') // Remove javascript: protocol
    .replace(/on\w+=/gi, '') // Remove event handlers
    .trim();
};

const sanitizeHtml = (html: string): string => {
  return html
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>/gi, '')
    .replace(/javascript:/gi, '')
    .replace(/on\w+=/gi, '');
};
```

### 2. XSS Prevention Patterns
```typescript
// Safe text rendering
const SafeText = ({ children }: { children: string }) => {
  const sanitizedText = sanitizeInput(children);
  return <Text>{sanitizedText}</Text>;
};

// Safe HTML rendering with DOMPurify
import DOMPurify from 'dompurify';

const SafeHtml = ({ html }: { html: string }) => {
  const sanitizedHtml = DOMPurify.sanitize(html);
  return <WebView source={{ html: sanitizedHtml }} />;
};
```

### 3. Form Validation & Sanitization
```typescript
// Secure form handling
interface SecureFormData {
  username: string;
  email: string;
  message: string;
}

const validateAndSanitizeForm = (data: SecureFormData): SecureFormData => {
  return {
    username: sanitizeInput(data.username).substring(0, 50),
    email: sanitizeInput(data.email).toLowerCase(),
    message: sanitizeInput(data.message).substring(0, 1000)
  };
};

const SecureForm = () => {
  const [formData, setFormData] = useState<SecureFormData>({
    username: '',
    email: '',
    message: ''
  });

  const handleSubmit = (data: SecureFormData) => {
    const sanitizedData = validateAndSanitizeForm(data);
    // Submit sanitized data
  };

  return (
    <Form onSubmit={handleSubmit}>
      <TextInput
        value={formData.username}
        onChangeText={(text) => setFormData(prev => ({
          ...prev,
          username: sanitizeInput(text)
        }))}
        maxLength={50}
      />
    </Form>
  );
};
```

### 4. URL Validation & Sanitization
```typescript
// Secure URL handling
const isValidUrl = (url: string): boolean => {
  try {
    const parsed = new URL(url);
    return ['http:', 'https:'].includes(parsed.protocol);
  } catch {
    return false;
  }
};

const SafeLink = ({ href, children }: { href: string; children: string }) => {
  if (!isValidUrl(href)) {
    return <Text>{children}</Text>;
  }
  
  return (
    <TouchableOpacity onPress={() => Linking.openURL(href)}>
      <Text>{children}</Text>
    </TouchableOpacity>
  );
};
```

### 5. Secure Data Storage
```typescript
// Secure AsyncStorage wrapper
import AsyncStorage from '@react-native-async-storage/async-storage';
import { encrypt, decrypt } from 'react-native-crypto';

const SecureStorage = {
  async setItem(key: string, value: string): Promise<void> {
    const encryptedValue = await encrypt(value);
    await AsyncStorage.setItem(key, encryptedValue);
  },

  async getItem(key: string): Promise<string | null> {
    const encryptedValue = await AsyncStorage.getItem(key);
    if (!encryptedValue) return null;
    
    return await decrypt(encryptedValue);
  }
};
```

### 6. API Security
```typescript
// Secure API client
const secureApiClient = {
  async post(endpoint: string, data: any) {
    const sanitizedData = JSON.parse(
      JSON.stringify(data, (key, value) => {
        if (typeof value === 'string') {
          return sanitizeInput(value);
        }
        return value;
      })
    );

    return fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': await SecureStorage.getItem('apiKey')
      },
      body: JSON.stringify(sanitizedData)
    });
  }
};
```

### 7. Content Security Policy
```typescript
// WebView with CSP
const SecureWebView = ({ html }: { html: string }) => {
  const cspHtml = `
    <!DOCTYPE html>
    <html>
      <head>
        <meta http-equiv="Content-Security-Policy" 
              content="default-src 'self'; script-src 'none'; style-src 'self' 'unsafe-inline';">
      </head>
      <body>${sanitizeHtml(html)}</body>
    </html>
  `;

  return <WebView source={{ html: cspHtml }} />;
};
```

### 8. Input Validation Hooks
```typescript
// Custom hooks for input validation
const useSecureInput = (initialValue: string = '') => {
  const [value, setValue] = useState(initialValue);
  const [isValid, setIsValid] = useState(true);

  const handleChange = (text: string) => {
    const sanitized = sanitizeInput(text);
    setValue(sanitized);
    setIsValid(sanitized.length > 0);
  };

  return { value, isValid, handleChange };
};

const useEmailValidation = () => {
  const [email, setEmail] = useState('');
  const [isValid, setIsValid] = useState(false);

  const validateEmail = (email: string) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(sanitizeInput(email));
  };

  const handleEmailChange = (text: string) => {
    const sanitized = sanitizeInput(text);
    setEmail(sanitized);
    setIsValid(validateEmail(sanitized));
  };

  return { email, isValid, handleEmailChange };
};
```

### 9. Secure Component Patterns
```typescript
// Secure text input component
const SecureTextInput = ({ 
  value, 
  onChangeText, 
  placeholder,
  maxLength = 100 
}: {
  value: string;
  onChangeText: (text: string) => void;
  placeholder: string;
  maxLength?: number;
}) => {
  const handleChange = (text: string) => {
    const sanitized = sanitizeInput(text).substring(0, maxLength);
    onChangeText(sanitized);
  };

  return (
    <TextInput
      value={value}
      onChangeText={handleChange}
      placeholder={sanitizeInput(placeholder)}
      maxLength={maxLength}
    />
  );
};

// Secure display component
const SecureDisplay = ({ content }: { content: string }) => {
  const sanitizedContent = sanitizeInput(content);
  
  return (
    <Text selectable={false}>
      {sanitizedContent}
    </Text>
  );
};
```

### 10. Error Handling Security
```typescript
// Secure error handling
const SecureErrorBoundary = ({ children }: { children: React.ReactNode }) => {
  const [hasError, setHasError] = useState(false);

  if (hasError) {
    return (
      <View>
        <Text>Something went wrong. Please try again.</Text>
      </View>
    );
  }

  return (
    <ErrorBoundary onError={() => setHasError(true)}>
      {children}
    </ErrorBoundary>
  );
};
```

## Security Checklist

### Input Validation
- [ ] Sanitize all user inputs
- [ ] Validate email formats
- [ ] Limit input lengths
- [ ] Remove dangerous characters
- [ ] Validate URLs before opening

### Data Storage
- [ ] Encrypt sensitive data
- [ ] Use secure storage APIs
- [ ] Clear data on logout
- [ ] Validate stored data integrity

### Network Security
- [ ] Use HTTPS only
- [ ] Validate API responses
- [ ] Sanitize data before sending
- [ ] Implement rate limiting

### UI Security
- [ ] Prevent XSS in text displays
- [ ] Sanitize HTML content
- [ ] Use secure WebView configurations
- [ ] Implement Content Security Policy

### Error Handling
- [ ] Don't expose sensitive information in errors
- [ ] Log errors securely
- [ ] Implement proper error boundaries
- [ ] Sanitize error messages

## Implementation Example

```typescript
// Complete secure form implementation
const SecureUserProfile = () => {
  const { value: username, isValid: isUsernameValid, handleChange: handleUsernameChange } = useSecureInput();
  const { email, isValid: isEmailValid, handleEmailChange } = useEmailValidation();
  const { value: bio, handleChange: handleBioChange } = useSecureInput();

  const handleSubmit = async () => {
    if (!isUsernameValid || !isEmailValid) {
      Alert.alert('Please fix validation errors');
      return;
    }

    const userData = {
      username: sanitizeInput(username),
      email: sanitizeInput(email),
      bio: sanitizeInput(bio)
    };

    try {
      await secureApiClient.post('/api/users', userData);
      Alert.alert('Profile updated successfully');
    } catch (error) {
      Alert.alert('Update failed. Please try again.');
    }
  };

  return (
    <SecureErrorBoundary>
      <View>
        <SecureTextInput
          value={username}
          onChangeText={handleUsernameChange}
          placeholder="Enter username"
          maxLength={30}
        />
        
        <SecureTextInput
          value={email}
          onChangeText={handleEmailChange}
          placeholder="Enter email"
          maxLength={100}
        />
        
        <SecureTextInput
          value={bio}
          onChangeText={handleBioChange}
          placeholder="Enter bio"
          maxLength={500}
        />
        
        <TouchableOpacity 
          onPress={handleSubmit}
          disabled={!isUsernameValid || !isEmailValid}
        >
          <Text>Update Profile</Text>
        </TouchableOpacity>
      </View>
    </SecureErrorBoundary>
  );
};
```

## Security Best Practices Summary

1. **Always sanitize user inputs** before processing or displaying
2. **Validate data formats** (email, URL, etc.) before use
3. **Use secure storage** for sensitive information
4. **Implement proper error handling** without exposing sensitive data
5. **Use HTTPS** for all network communications
6. **Limit input lengths** to prevent buffer overflow attacks
7. **Remove dangerous characters** and scripts from user content
8. **Implement Content Security Policy** for WebView content
9. **Use secure authentication** and session management
10. **Regular security audits** and penetration testing

This pattern ensures comprehensive security for React Native applications while maintaining good user experience and performance. 