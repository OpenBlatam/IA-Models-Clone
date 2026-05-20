from typing import Dict

class ExpoGenerator:
    """
    Generates Expo React Native application code.
    """
    
    def generate_project_structure(self, project_name: str) -> Dict[str, str]:
        """
        Returns a dictionary representing the file structure of an Expo app.
        """
        structure = {
            f"{project_name}/package.json": self._generate_package_json(project_name),
            f"{project_name}/app.json": self._generate_app_json(project_name),
            f"{project_name}/App.tsx": self._generate_app_tsx(),
            f"{project_name}/components/Screen.tsx": self.generate_screen("Home", "HomeScreen"),
        }
        return structure

    def generate_screen(self, prompt: str, name: str) -> str:
        """
        Generates a React Native screen component.
        """
        return f"""import React from 'react';
import {{ StyleSheet, Text, View, SafeAreaView }} from 'react-native';

export default function {name}() {{
  return (
    <SafeAreaView style={{styles.container}}>
      <View style={{styles.content}}>
        <Text style={{styles.title}}>{prompt}</Text>
        <Text style={{styles.text}}>Generated screen for: {prompt}</Text>
      </View>
    </SafeAreaView>
  );
}}

const styles = StyleSheet.create({{
  container: {{
    flex: 1,
    backgroundColor: '#fff',
  }},
  content: {{
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  }},
  title: {{
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 10,
  }},
  text: {{
    fontSize: 16,
    color: '#666',
  }},
}});
"""

    def generate_component(self, prompt: str, name: str) -> str:
        """
        Generates a React Native component.
        """
        return f"""import React from 'react';
import {{ StyleSheet, View, Text }} from 'react-native';

interface {name}Props {{
  title: string;
}}

export const {name}: React.FC<{name}Props> = ({{ title }}) => {{
  return (
    <View style={{styles.card}}>
      <Text style={{styles.title}}>{{title}}</Text>
    </View>
  );
}};

const styles = StyleSheet.create({{
  card: {{
    padding: 15,
    borderRadius: 8,
    backgroundColor: '#f8f9fa',
    marginVertical: 10,
  }},
  title: {{
    fontSize: 18,
    fontWeight: '600',
  }},
}});
"""

    def _generate_package_json(self, name: str) -> str:
        return f"""{{
  "name": "{name}",
  "version": "1.0.0",
  "main": "node_modules/expo/AppEntry.js",
  "scripts": {{
    "start": "expo start",
    "android": "expo start --android",
    "ios": "expo start --ios",
    "web": "expo start --web"
  }},
  "dependencies": {{
    "expo": "~50.0.0",
    "expo-status-bar": "~1.11.1",
    "react": "18.2.0",
    "react-native": "0.73.2"
  }},
  "devDependencies": {{
    "@babel/core": "^7.20.0",
    "@types/react": "~18.2.45",
    "typescript": "^5.1.3"
  }},
  "private": true
}}"""

    def _generate_app_json(self, name: str) -> str:
        return f"""{{
  "expo": {{
    "name": "{name}",
    "slug": "{name}",
    "version": "1.0.0",
    "orientation": "portrait",
    "icon": "./assets/icon.png",
    "userInterfaceStyle": "light",
    "splash": {{
      "image": "./assets/splash.png",
      "resizeMode": "contain",
      "backgroundColor": "#ffffff"
    }},
    "assetBundlePatterns": [
      "**/*"
    ],
    "ios": {{
      "supportsTablet": true
    }},
    "android": {{
      "adaptiveIcon": {{
        "foregroundImage": "./assets/adaptive-icon.png",
        "backgroundColor": "#ffffff"
      }}
    }},
    "web": {{
      "favicon": "./assets/favicon.png"
    }}
  }}
}}"""

    def _generate_app_tsx(self) -> str:
        return """import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View } from 'react-native';

export default function App() {
  return (
    <View style={styles.container}>
      <Text>Open up App.tsx to start working on your app!</Text>
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
"""
