# Assets Required

The following assets are referenced in `app.json` and need to be added to the `assets/` folder:

## Required Assets

1. **icon.png** (1024x1024px)
   - App icon for iOS and Android
   - Should be square with no transparency

2. **splash.png** (1284x2778px recommended)
   - Splash screen image
   - Should match your app's branding

3. **adaptive-icon.png** (Android)
   - Foreground: 1024x1024px
   - Background: 1024x1024px
   - Used for Android adaptive icons

4. **favicon.png** (48x48px)
   - Web favicon

## Creating Assets

You can use tools like:
- [Expo Asset Generator](https://www.npmjs.com/package/@expo/asset-generator)
- [App Icon Generator](https://www.appicon.co/)
- Design tools like Figma, Sketch, or Adobe XD

## Quick Setup

1. Create the `assets/` folder in the root directory
2. Add your icon and splash images
3. Update `app.json` if your file names differ

## Temporary Solution

For development, you can use placeholder images or Expo's default assets by removing the asset references from `app.json` temporarily.


