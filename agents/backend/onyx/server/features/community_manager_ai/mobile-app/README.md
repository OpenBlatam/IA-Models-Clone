# Community Manager AI - Mobile App

A comprehensive React Native mobile application built with Expo and TypeScript for managing social media content across multiple platforms.

## 🚀 Features

- **Dashboard**: Overview of posts, engagement metrics, and upcoming content
- **Posts Management**: Create, schedule, publish, and manage posts across platforms
  - ✅ Full form validation with Zod
  - ✅ Multi-platform selection
  - ✅ Date/time scheduling
  - ✅ Tags management
- **Calendar View**: Visual calendar with scheduled posts and events
- **Memes Library**: Upload, organize, and manage meme content
- **Platform Integration**: Connect and manage multiple social media platforms
  - Facebook
  - Instagram
  - Twitter/X
  - LinkedIn
  - TikTok
  - YouTube
- **Analytics**: Track engagement, performance metrics, and trends
- **Templates**: Create and manage reusable content templates
  - ✅ Automatic variable detection
  - ✅ Template creation form
  - ✅ Category management

## 📋 Prerequisites

- Node.js 18+ and npm/yarn
- Expo CLI (`npm install -g expo-cli`)
- iOS Simulator (for iOS development) or Android Emulator (for Android development)
- Backend API running (see backend documentation)

## 🛠️ Installation

1. Navigate to the mobile app directory:
```bash
cd mobile-app
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Configure API endpoint:
   - Update `app.json` or create `.env` file with your backend API URL:
   ```
   EXPO_PUBLIC_API_URL=http://localhost:8000
   ```

## 🏃 Running the App

### Development Mode

```bash
npm start
# or
yarn start
```

This will start the Expo development server. You can then:
- Press `i` to open iOS simulator
- Press `a` to open Android emulator
- Scan QR code with Expo Go app on your physical device

### Building for Production

#### iOS
```bash
npm run ios
# or
expo build:ios
```

#### Android
```bash
npm run android
# or
expo build:android
```

## 📱 App Structure

```
mobile-app/
├── app/                    # Expo Router pages
│   ├── (tabs)/            # Tab navigation screens
│   │   ├── dashboard.tsx
│   │   ├── posts.tsx
│   │   ├── calendar.tsx
│   │   ├── memes.tsx
│   │   ├── platforms.tsx
│   │   ├── analytics.tsx
│   │   └── templates.tsx
│   ├── _layout.tsx        # Root layout
│   ├── index.tsx         # Entry point
│   └── login.tsx         # Login screen
├── components/            # Reusable components
├── hooks/                # Custom React hooks
│   └── useApi.ts        # API hooks with React Query
├── lib/                  # Utilities and services
│   └── api.ts           # API client
├── store/                # State management
│   └── useAuthStore.ts  # Auth state (Zustand)
├── types/                # TypeScript types
│   └── index.ts
└── utils/                # Helper functions
```

## 🔧 Configuration

### API Configuration

The app connects to the backend API. Update the API URL in:
- `lib/api.ts` - Default API client configuration
- Or set `EXPO_PUBLIC_API_URL` environment variable

### Authentication

The app uses secure token storage via `expo-secure-store`. Tokens are automatically included in API requests.

## 📚 Key Technologies

- **Expo**: React Native framework
- **Expo Router**: File-based routing
- **TypeScript**: Type safety
- **React Query**: Data fetching and caching
- **Zustand**: Lightweight state management
- **React Hook Form**: Form management
- **Zod**: Schema validation
- **React Native Calendars**: Calendar component
- **React Native Chart Kit**: Analytics charts
- **DateTimePicker**: Date/time selection
- **NativeWind**: Tailwind CSS for React Native

## 🎨 UI/UX Features

- Modern, clean interface
- Dark mode support (automatic)
- Responsive design
- Pull-to-refresh on all lists
- Loading states and error handling
- Toast notifications
- Form validation with real-time feedback
- Empty states with call-to-action
- Keyboard-aware scrolling
- Reusable UI components

## 🔐 Security

- Secure token storage using `expo-secure-store`
- HTTPS API communication
- Input validation
- Error handling and user feedback

## 📝 API Integration

The app integrates with the Community Manager AI backend API:

- **Posts**: `/posts` - Create, read, update, delete posts
- **Memes**: `/memes` - Upload and manage memes
- **Calendar**: `/calendar` - Get scheduled events
- **Platforms**: `/platforms` - Connect/disconnect platforms
- **Analytics**: `/analytics` - Get performance metrics
- **Templates**: `/templates` - Manage content templates
- **Dashboard**: `/dashboard` - Get overview data

See `lib/api.ts` for complete API client implementation.

## 🧪 Testing

```bash
npm test
# or
yarn test
```

## 📦 Building

### EAS Build (Recommended)

1. Install EAS CLI:
```bash
npm install -g eas-cli
```

2. Configure:
```bash
eas build:configure
```

3. Build:
```bash
eas build --platform ios
eas build --platform android
```

## 🐛 Troubleshooting

### Common Issues

1. **Metro bundler cache issues**:
```bash
npm start -- --clear
```

2. **Node modules issues**:
```bash
rm -rf node_modules
npm install
```

3. **iOS build issues**:
```bash
cd ios
pod install
cd ..
```

## 📄 License

See main project LICENSE file.

## 🤝 Contributing

1. Follow TypeScript best practices
2. Use functional components
3. Follow the existing code style
4. Add proper error handling
5. Test on both iOS and Android

## 📞 Support

For issues and questions, please refer to the main project documentation or create an issue in the repository.

