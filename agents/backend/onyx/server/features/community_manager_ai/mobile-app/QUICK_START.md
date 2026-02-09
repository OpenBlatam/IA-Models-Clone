# Quick Start Guide - Mobile App

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
cd mobile-app
npm install
```

### Step 2: Configure API Endpoint

Create a `.env` file in the `mobile-app` directory:

```env
EXPO_PUBLIC_API_URL=http://localhost:8000
```

Or update the API URL directly in `lib/api.ts`:

```typescript
const API_URL = 'http://your-backend-url:8000';
```

### Step 3: Start the Development Server

```bash
npm start
```

### Step 4: Run on Device/Simulator

- **iOS Simulator**: Press `i` in the terminal
- **Android Emulator**: Press `a` in the terminal
- **Physical Device**: Scan the QR code with Expo Go app

## 📱 App Features Overview

### Dashboard
- View overview statistics
- See upcoming posts
- Monitor engagement metrics

### Posts
- Create new posts
- Schedule posts for multiple platforms
- Publish immediately or schedule for later
- Filter by status (All, Scheduled, Published)

### Calendar
- Visual calendar view
- See all scheduled posts
- View daily events

### Memes
- Upload memes from device
- Organize by category
- Search and filter memes

### Platforms
- Connect social media accounts
- Manage platform connections
- View connection status

### Analytics
- Platform-specific analytics
- Engagement metrics
- Best performing posts
- Trends visualization

### Templates
- Create reusable content templates
- Use variables in templates
- Search and manage templates

## 🔧 Common Commands

```bash
# Start development server
npm start

# Clear cache and start
npm start -- --clear

# Type checking
npm run type-check

# Run on iOS
npm run ios

# Run on Android
npm run android
```

## 🐛 Troubleshooting

### Issue: Metro bundler not starting
**Solution**: Clear cache
```bash
npm start -- --clear
```

### Issue: Can't connect to backend API
**Solution**: 
1. Ensure backend is running
2. Check API URL in `lib/api.ts`
3. Verify network connectivity

### Issue: Build errors
**Solution**: 
```bash
rm -rf node_modules
npm install
```

## 📚 Next Steps

1. **Customize Theme**: Edit colors in `utils/constants.ts`
2. **Add Features**: Extend screens in `app/(tabs)/`
3. **API Integration**: Update `lib/api.ts` for new endpoints
4. **Styling**: Modify styles in individual screen files

## 🔐 Authentication

The app currently uses a simple token-based authentication. For production:

1. Implement proper OAuth flow
2. Add refresh token mechanism
3. Secure token storage (already using `expo-secure-store`)

## 📦 Building for Production

### Using EAS Build (Recommended)

```bash
# Install EAS CLI
npm install -g eas-cli

# Login
eas login

# Configure
eas build:configure

# Build for iOS
eas build --platform ios

# Build for Android
eas build --platform android
```

## 🎨 Customization

### Change App Name
Edit `app.json`:
```json
{
  "expo": {
    "name": "Your App Name"
  }
}
```

### Change Colors
Edit `utils/constants.ts`:
```typescript
export const COLORS = {
  primary: '#your-color',
  // ...
}
```

## 📞 Need Help?

- Check the main README.md
- Review Expo documentation: https://docs.expo.dev/
- Check backend API documentation


