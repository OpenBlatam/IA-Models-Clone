# API Client Documentation

## Overview

The API client provides a robust, type-safe interface for communicating with the Music Analyzer API backend.

## Features

- ✅ Automatic retry logic with exponential backoff
- ✅ Health check monitoring
- ✅ Request/response logging (development only)
- ✅ Comprehensive error handling
- ✅ Type-safe requests and responses
- ✅ Request cancellation support
- ✅ Connection status monitoring

## Usage

### Basic API Calls

```typescript
import { searchTracks, analyzeTrack } from '@/lib/api';

// Search tracks
const results = await searchTracks('artist name', 10);

// Analyze a track
const analysis = await analyzeTrack({
  trackId: 'track-id',
  includeCoaching: true,
});
```

### Health Check

```typescript
import { checkApiHealth } from '@/lib/api';

const health = await checkApiHealth();
console.log(health.status); // 'healthy' | 'unhealthy'
```

### Connection Testing

```typescript
import { testApiConnection, validateApiConfig } from '@/lib/api';

// Test connection
const test = await testApiConnection();
console.log(test.success, test.responseTime);

// Validate configuration
const validation = validateApiConfig();
if (!validation.isValid) {
  console.error('Config issues:', validation.issues);
}
```

### Using React Query

```typescript
import { useQuery } from '@tanstack/react-query';
import { searchTracks } from '@/lib/api';
import { QUERY_KEYS } from '@/lib/constants';

const { data, isLoading, error } = useQuery({
  queryKey: QUERY_KEYS.MUSIC.SEARCH(query),
  queryFn: () => searchTracks(query, 10),
  enabled: query.length > 0,
});
```

### API Health Monitoring Hook

```typescript
import { useApiHealth } from '@/lib/hooks';

function MyComponent() {
  const { isHealthy, isLoading, message, refreshHealth } = useApiHealth({
    refetchInterval: 30000, // Check every 30 seconds
  });

  return (
    <div>
      Status: {isHealthy ? 'Connected' : 'Disconnected'}
      <button onClick={refreshHealth}>Refresh</button>
    </div>
  );
}
```

## Error Handling

The API client uses custom error types:

- `ApiError`: API errors with status codes
- `NetworkError`: Network connection issues
- `ValidationError`: Input validation errors

```typescript
import { ApiError, NetworkError, getErrorMessage } from '@/lib/api';

try {
  await searchTracks('query');
} catch (error) {
  if (error instanceof NetworkError) {
    console.error('Network issue:', error.message);
  } else if (error instanceof ApiError) {
    console.error('API error:', error.statusCode, error.message);
  } else {
    console.error('Unknown error:', getErrorMessage(error));
  }
}
```

## Configuration

API configuration is managed in `lib/config/app.ts`:

```typescript
export const apiConfig = {
  music: {
    baseURL: `${env.MUSIC_API_URL}/music`,
    timeout: 30000,
    retries: 2,
  },
};
```

## Environment Variables

Required environment variables:

```env
NEXT_PUBLIC_MUSIC_API_URL=http://localhost:8010
NEXT_PUBLIC_ROBOT_API_URL=http://localhost:8010
```

## Retry Logic

The client automatically retries failed requests with exponential backoff:

- Default: 2 retries
- Retry delay: 1000ms * 2^attempt
- Only retries on network errors or 5xx server errors

## Request Logging

In development mode, all requests and responses are logged to the console with:
- Request method and URL
- Request parameters and data
- Response status and duration
- Request ID for tracking

## Connection Status Component

Use the `ApiStatus` component to show connection status:

```typescript
import { ApiStatus } from '@/components/api-status';

<ApiStatus showDetails={true} position="top-right" />
```

