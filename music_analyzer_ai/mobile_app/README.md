# Music Analyzer AI - Mobile App

A React Native mobile application built with Expo and TypeScript that provides AI-powered music analysis capabilities.

## Features

- 🎵 **Search Music**: Search for tracks using Spotify integration
- 📊 **Music Analysis**: Get detailed musical analysis including:
  - Key signature and scale detection
  - Tempo and time signature analysis
  - Technical features (energy, danceability, valence, etc.)
  - Composition and structure analysis
- 🎓 **Coaching**: Receive personalized learning paths and practice exercises
- ⭐ **Favorites**: Save your favorite tracks
- 📱 **Modern UI**: Beautiful, responsive design with dark mode support

## Tech Stack

- **Expo** ~50.0.0
- **React Native** 0.73.0
- **TypeScript** ~5.3.0
- **Expo Router** for navigation
- **React Query** for data fetching and caching
- **Zustand** for state management
- **Zod** for validation
- **Axios** for API communication

## Installation

1. Install dependencies:
```bash
npm install
# or
yarn install
```

2. Configure API endpoint in `app.json`:
```json
{
  "expo": {
    "extra": {
      "apiUrl": "http://your-api-url:8010"
    }
  }
}
```

3. Start the development server:
```bash
npm start
# or
yarn start
```

## Project Structure

```
mobile_app/
├── src/
│   ├── app/              # Expo Router pages
│   ├── components/       # React components
│   │   ├── common/       # Reusable components
│   │   └── music/        # Music-specific components
│   ├── contexts/         # React contexts
│   ├── hooks/            # Custom hooks
│   ├── services/         # API services
│   ├── types/            # TypeScript types
│   ├── utils/            # Utility functions
│   └── constants/        # Constants and config
├── assets/               # Images, fonts, etc.
├── app.json              # Expo configuration
├── package.json
└── tsconfig.json
```

## Development

### Running on iOS
```bash
npm run ios
```

### Running on Android
```bash
npm run android
```

### Running on Web
```bash
npm run web
```

### Type Checking
```bash
npm run type-check
```

### Linting
```bash
npm run lint
```

## API Integration

The app connects to the Music Analyzer AI backend API. Make sure the backend server is running and accessible at the configured API URL.

### Main Endpoints Used:
- `POST /music/search` - Search for tracks
- `POST /music/analyze` - Analyze a track
- `GET /music/analyze/{track_id}` - Get analysis by track ID
- `GET /music/track/{track_id}/recommendations` - Get recommendations
- `GET /music/health` - Health check

## Features in Detail

### Search Screen
- Real-time search with debouncing
- Track preview cards
- Recent searches history
- Error handling and retry

### Analysis Screen
- Comprehensive musical analysis display
- Technical features visualization
- Coaching recommendations
- Learning paths and exercises

### State Management
- React Context for global music state
- React Query for server state
- Local storage for favorites and history

## Best Practices

- ✅ TypeScript strict mode enabled
- ✅ Functional components with hooks
- ✅ Error boundaries for crash prevention
- ✅ Proper loading and error states
- ✅ Accessibility support
- ✅ Safe area handling for all devices
- ✅ Performance optimizations (memoization, code splitting)

## License

MIT

