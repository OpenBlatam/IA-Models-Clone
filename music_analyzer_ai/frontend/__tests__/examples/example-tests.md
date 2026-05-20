# Example Tests - Reference Guide

## 📚 Ejemplos Completos de Tests

### 1. Test de Componente Simple

```typescript
import { render, screen } from '@testing-library/react';
import { Component } from '@/components/Component';

describe('Component', () => {
  it('should render correctly', () => {
    render(<Component />);
    expect(screen.getByText('Hello')).toBeInTheDocument();
  });
});
```

### 2. Test con Props

```typescript
import { render, screen } from '@testing-library/react';
import { Component } from '@/components/Component';

describe('Component', () => {
  it('should render with props', () => {
    render(<Component name="Test" />);
    expect(screen.getByText('Test')).toBeInTheDocument();
  });
});
```

### 3. Test con QueryClient

```typescript
import { renderWithQueryClient } from '@/__tests__/setup/test-utils';
import { Component } from '@/components/Component';

describe('Component', () => {
  it('should work with QueryClient', () => {
    renderWithQueryClient(<Component />);
    // ...
  });
});
```

### 4. Test con User Events

```typescript
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Component } from '@/components/Component';

describe('Component', () => {
  it('should handle click', async () => {
    const user = userEvent.setup();
    render(<Component />);
    
    const button = screen.getByRole('button');
    await user.click(button);
    
    expect(screen.getByText('Clicked')).toBeInTheDocument();
  });
});
```

### 5. Test con Mocks

```typescript
import { render, screen, waitFor } from '@testing-library/react';
import { Component } from '@/components/Component';
import { musicApiService } from '@/lib/api/music-api';

jest.mock('@/lib/api/music-api');

describe('Component', () => {
  it('should load data', async () => {
    (musicApiService.searchTracks as jest.Mock).mockResolvedValue({
      success: true,
      results: [],
    });

    render(<Component />);
    
    await waitFor(() => {
      expect(musicApiService.searchTracks).toHaveBeenCalled();
    });
  });
});
```

### 6. Test con Store

```typescript
import { render, screen } from '@testing-library/react';
import { useMusicStore } from '@/lib/store/music-store';
import { Component } from '@/components/Component';
import { act } from 'react';

describe('Component', () => {
  it('should use store', () => {
    render(<Component />);
    
    act(() => {
      useMusicStore.getState().setCurrentTrack(mockTrack);
    });
    
    expect(screen.getByText('Test Track')).toBeInTheDocument();
  });
});
```

### 7. Test con Async Operations

```typescript
import { render, screen, waitFor } from '@testing-library/react';
import { Component } from '@/components/Component';

describe('Component', () => {
  it('should load async data', async () => {
    render(<Component />);
    
    expect(screen.getByText('Loading...')).toBeInTheDocument();
    
    await waitFor(() => {
      expect(screen.getByText('Loaded')).toBeInTheDocument();
    });
  });
});
```

### 8. Test con Error Handling

```typescript
import { render, screen } from '@testing-library/react';
import { Component } from '@/components/Component';
import { musicApiService } from '@/lib/api/music-api';

jest.mock('@/lib/api/music-api');

describe('Component', () => {
  it('should handle errors', async () => {
    (musicApiService.searchTracks as jest.Mock).mockRejectedValue(
      new Error('API Error')
    );

    render(<Component />);
    
    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument();
    });
  });
});
```

### 9. Test con Formularios

```typescript
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FormComponent } from '@/components/FormComponent';

describe('FormComponent', () => {
  it('should submit form', async () => {
    const user = userEvent.setup();
    const onSubmit = jest.fn();
    
    render(<FormComponent onSubmit={onSubmit} />);
    
    const input = screen.getByLabelText('Name');
    await user.type(input, 'Test');
    
    const button = screen.getByRole('button', { name: /submit/i });
    await user.click(button);
    
    expect(onSubmit).toHaveBeenCalledWith({ name: 'Test' });
  });
});
```

### 10. Test con Timers

```typescript
import { render, screen } from '@testing-library/react';
import { Component } from '@/components/Component';

describe('Component', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should debounce input', async () => {
    render(<Component />);
    
    const input = screen.getByRole('textbox');
    await user.type(input, 'test');
    
    jest.advanceTimersByTime(500);
    
    await waitFor(() => {
      expect(screen.getByText('test')).toBeInTheDocument();
    });
  });
});
```

## 🎯 Patrones Comunes

### Setup y Teardown

```typescript
describe('Component', () => {
  beforeEach(() => {
    // Setup antes de cada test
    jest.clearAllMocks();
  });

  afterEach(() => {
    // Cleanup después de cada test
    cleanup();
  });
});
```

### Mocks Globales

```typescript
// En jest.setup.js
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: jest.fn(),
  }),
}));
```

### Helpers Personalizados

```typescript
// En test-utils.tsx
export function renderWithProviders(ui: ReactElement) {
  return render(ui, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={createTestQueryClient()}>
        {children}
      </QueryClientProvider>
    ),
  });
}
```

## ✨ Mejores Prácticas

1. ✅ Usar AAA Pattern (Arrange, Act, Assert)
2. ✅ Tests independientes
3. ✅ Nombres descriptivos
4. ✅ Casos edge
5. ✅ Mocks apropiados
6. ✅ Async handling correcto
7. ✅ Cleanup después de tests

