# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies
```bash
npm install
```

### Step 2: Start the Backend
Make sure the backend API is running:
```bash
# In the backend directory
cd ../../ai_job_replacement_helper
python main.py
```

The API should be running on `http://localhost:8030`

### Step 3: Start the Mobile App
```bash
npm start
```

### Step 4: Run on Your Device
- **iOS**: Press `i` in the terminal or scan QR code with Camera app
- **Android**: Press `a` in the terminal or scan QR code with Expo Go app
- **Web**: Press `w` in the terminal

## 📱 First Time Setup

1. **Create an Account**
   - Open the app
   - Tap "Sign Up"
   - Enter your email, username, and password
   - You'll be automatically logged in

2. **Explore Features**
   - **Dashboard**: See your progress and stats
   - **Jobs**: Swipe through job listings (Tinder-style)
   - **Roadmap**: Follow the step-by-step career guide
   - **Profile**: View your badges and settings

## 🔧 Configuration

### Change API URL
Edit `src/constants/config.ts`:
```typescript
export const API_BASE_URL = 'http://your-api-url:8030';
```

### For Physical Device Testing
Use your computer's IP address instead of `localhost`:
```typescript
export const API_BASE_URL = 'http://192.168.1.100:8030';
```

## 🐛 Common Issues

### "Network Error"
- Ensure backend is running
- Check API URL in config
- For physical devices, use IP address not localhost

### "Module not found"
```bash
rm -rf node_modules
npm install
```

### "Expo Go not working"
- Update Expo Go app on your device
- Clear cache: `expo start -c`

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [ASSETS.md](ASSETS.md) for asset requirements
- Explore the codebase structure in the README

## 💡 Tips

- Use React Query DevTools for debugging API calls
- Enable remote debugging for better error messages
- Check Expo logs for detailed error information


