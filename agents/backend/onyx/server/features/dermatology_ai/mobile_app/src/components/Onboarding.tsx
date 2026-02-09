import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, { useSharedValue, useAnimatedStyle, withSpring } from 'react-native-reanimated';

interface OnboardingStep {
  id: number;
  title: string;
  description: string;
  icon: keyof typeof Ionicons.glyphMap;
  color: string;
}

interface OnboardingProps {
  steps: OnboardingStep[];
  onComplete: () => void;
  onSkip?: () => void;
}

const Onboarding: React.FC<OnboardingProps> = ({
  steps,
  onComplete,
  onSkip,
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const screenWidth = Dimensions.get('window').width;
  const translateX = useSharedValue(0);

  const animatedStyle = useAnimatedStyle(() => {
    return {
      transform: [{ translateX: translateX.value }],
    };
  });

  const goToNext = () => {
    if (currentStep < steps.length - 1) {
      const nextStep = currentStep + 1;
      setCurrentStep(nextStep);
      translateX.value = withSpring(-nextStep * screenWidth);
    } else {
      onComplete();
    }
  };

  const goToPrevious = () => {
    if (currentStep > 0) {
      const prevStep = currentStep - 1;
      setCurrentStep(prevStep);
      translateX.value = withSpring(-prevStep * screenWidth);
    }
  };

  const goToStep = (index: number) => {
    setCurrentStep(index);
    translateX.value = withSpring(-index * screenWidth);
  };

  return (
    <View style={styles.container}>
      {onSkip && (
        <TouchableOpacity style={styles.skipButton} onPress={onSkip}>
          <Text style={styles.skipText}>Omitir</Text>
        </TouchableOpacity>
      )}

      <Animated.View style={[styles.slider, animatedStyle]}>
        {steps.map((step, index) => (
          <View
            key={step.id}
            style={[styles.step, { width: screenWidth }]}
          >
            <LinearGradient
              colors={[step.color, `${step.color}80`]}
              style={styles.iconContainer}
            >
              <Ionicons name={step.icon} size={64} color="#fff" />
            </LinearGradient>
            <Text style={styles.title}>{step.title}</Text>
            <Text style={styles.description}>{step.description}</Text>
          </View>
        ))}
      </Animated.View>

      <View style={styles.indicators}>
        {steps.map((_, index) => (
          <TouchableOpacity
            key={index}
            style={[
              styles.indicator,
              index === currentStep && styles.indicatorActive,
            ]}
            onPress={() => goToStep(index)}
          />
        ))}
      </View>

      <View style={styles.buttons}>
        {currentStep > 0 && (
          <TouchableOpacity
            style={styles.prevButton}
            onPress={goToPrevious}
          >
            <Ionicons name="chevron-back" size={24} color="#6366f1" />
            <Text style={styles.prevButtonText}>Anterior</Text>
          </TouchableOpacity>
        )}
        <TouchableOpacity
          style={styles.nextButton}
          onPress={goToNext}
        >
          <Text style={styles.nextButtonText}>
            {currentStep === steps.length - 1 ? 'Comenzar' : 'Siguiente'}
          </Text>
          <Ionicons name="chevron-forward" size={24} color="#fff" />
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  skipButton: {
    position: 'absolute',
    top: 50,
    right: 20,
    zIndex: 10,
    padding: 8,
  },
  skipText: {
    fontSize: 16,
    color: '#6366f1',
    fontWeight: '600',
  },
  slider: {
    flex: 1,
    flexDirection: 'row',
  },
  step: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  iconContainer: {
    width: 120,
    height: 120,
    borderRadius: 60,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 32,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 16,
    textAlign: 'center',
  },
  description: {
    fontSize: 16,
    color: '#6b7280',
    textAlign: 'center',
    lineHeight: 24,
  },
  indicators: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 20,
  },
  indicator: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#d1d5db',
    marginHorizontal: 4,
  },
  indicatorActive: {
    width: 24,
    backgroundColor: '#6366f1',
  },
  buttons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 20,
    paddingBottom: 40,
  },
  prevButton: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
  },
  prevButtonText: {
    fontSize: 16,
    color: '#6366f1',
    fontWeight: '600',
    marginLeft: 4,
  },
  nextButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#6366f1',
    paddingHorizontal: 24,
    paddingVertical: 16,
    borderRadius: 12,
  },
  nextButtonText: {
    fontSize: 16,
    color: '#fff',
    fontWeight: '600',
    marginRight: 8,
  },
});

export default Onboarding;

