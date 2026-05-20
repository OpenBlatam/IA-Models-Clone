# Deployment Guide

## 🚀 Building for Production

### Prerequisites

1. Install EAS CLI:
```bash
npm install -g eas-cli
```

2. Login to Expo:
```bash
eas login
```

3. Configure project:
```bash
eas build:configure
```

### Building

#### iOS
```bash
eas build --platform ios
```

#### Android
```bash
eas build --platform android
```

#### Both Platforms
```bash
eas build --platform all
```

### Build Profiles

The `eas.json` file contains three build profiles:

1. **development** - Development builds with dev client
2. **preview** - Internal distribution (APK for Android)
3. **production** - Production builds for app stores

### Submitting to App Stores

#### iOS (App Store)
```bash
eas submit --platform ios
```

#### Android (Google Play)
```bash
eas submit --platform android
```

## 📱 OTA Updates

### Publishing Updates
```bash
eas update --branch production --message "Bug fixes"
```

### Update Channels
- `production` - Production updates
- `preview` - Preview updates
- `development` - Development updates

## 🔐 Environment Variables

Create `.env` file:
```env
EXPO_PUBLIC_API_URL=https://api.example.com
```

Or use EAS Secrets:
```bash
eas secret:create --scope project --name EXPO_PUBLIC_API_URL --value https://api.example.com
```

## 📦 App Store Requirements

### iOS
- App Store Connect account
- Apple Developer account ($99/year)
- App icons and screenshots
- Privacy policy URL
- App description and metadata

### Android
- Google Play Console account ($25 one-time)
- App icons and screenshots
- Privacy policy URL
- App description and metadata

## 🎯 Pre-Deployment Checklist

- [ ] Update version in `app.json`
- [ ] Update build number
- [ ] Test on physical devices
- [ ] Test all features
- [ ] Check for console errors
- [ ] Verify API endpoints
- [ ] Test offline mode
- [ ] Verify permissions
- [ ] Check accessibility
- [ ] Review security
- [ ] Update documentation
- [ ] Prepare app store assets

## 📊 Monitoring

After deployment, monitor:
- Crash reports (Sentry)
- Analytics (Firebase, Mixpanel)
- Performance metrics
- User feedback
- App store reviews

## 🔄 Continuous Deployment

Set up CI/CD with:
- GitHub Actions
- GitLab CI
- CircleCI
- Or similar

Example workflow:
1. Push to main branch
2. Run tests
3. Build app
4. Submit to stores (optional)
5. Publish OTA update

## 📝 Version Management

Update version in `app.json`:
```json
{
  "expo": {
    "version": "1.0.0",
    "ios": {
      "buildNumber": "1"
    },
    "android": {
      "versionCode": 1
    }
  }
}
```

## 🎉 Post-Deployment

1. Monitor crash reports
2. Track user analytics
3. Respond to reviews
4. Plan next update
5. Gather user feedback

