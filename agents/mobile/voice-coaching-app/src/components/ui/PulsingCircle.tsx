import React, { useRef, useEffect } from "react";
import { Animated } from "react-native";
/// <reference types="nativewind/types" />
import { COLORS } from "../../constants";

export interface PulsingCircleProps {
    isActive: boolean;
    size?: number;
    color?: string;
}

/**
 * Animated pulsing circle for recording indicator
 */
export function PulsingCircle({
    isActive,
    size = 256,
    color = COLORS.primary,
}: PulsingCircleProps) {
    const scaleAnim = useRef(new Animated.Value(1)).current;
    const opacityAnim = useRef(new Animated.Value(0.5)).current;

    useEffect(() => {
        let loop: Animated.CompositeAnimation | null = null;
        if (isActive) {
            loop = Animated.loop(
                Animated.parallel([
                    Animated.sequence([
                        Animated.timing(scaleAnim, {
                            toValue: 1.3,
                            duration: 1000,
                            useNativeDriver: true,
                        }),
                        Animated.timing(scaleAnim, {
                            toValue: 1,
                            duration: 1000,
                            useNativeDriver: true,
                        }),
                    ]),
                    Animated.sequence([
                        Animated.timing(opacityAnim, {
                            toValue: 0.2,
                            duration: 1000,
                            useNativeDriver: true,
                        }),
                        Animated.timing(opacityAnim, {
                            toValue: 0.5,
                            duration: 1000,
                            useNativeDriver: true,
                        }),
                    ]),
                ])
            );
            loop.start();
        } else {
            scaleAnim.setValue(1);
            opacityAnim.setValue(0.5);
        }

        return () => {
            if (loop) {
                loop.stop();
            }
        };
    }, [isActive, scaleAnim, opacityAnim]);

    return (
        <Animated.View
            style={{
                position: "absolute",
                width: size,
                height: size,
                borderRadius: size / 2,
                backgroundColor: color,
                transform: [{ scale: scaleAnim }],
                opacity: opacityAnim,
            }}
        />
    );
}
