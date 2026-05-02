import React from "react";
import { View, Text, TouchableOpacity } from "react-native";
/// <reference types="nativewind/types" />
import { haptics } from "../../utils/haptics";
import { formatRecordingTime } from "../../utils/dateUtils";
import { PulsingCircle } from "./PulsingCircle";

interface RecordingButtonProps {
    isRecording: boolean;
    recordingTime: number;
    onPress: () => void;
    disabled?: boolean;
}

/**
 * Large circular recording button
 */
export function RecordingButton({
    isRecording,
    recordingTime,
    onPress,
    disabled = false,
}: RecordingButtonProps) {
    const handlePress = async () => {
        await haptics.medium();
        onPress();
    };

    return (
        <View className="relative items-center justify-center">
            <PulsingCircle isActive={isRecording} />
            <TouchableOpacity
                onPress={handlePress}
                disabled={disabled}
                activeOpacity={0.8}
                className={`w-48 h-48 rounded-full items-center justify-center z-10 ${isRecording ? "bg-secondary" : "bg-primary"
                    }`}
            >
                {isRecording ? (
                    <View className="items-center">
                        <Text className="text-5xl">🎙️</Text>
                        <Text className="text-white text-2xl font-bold mt-2">
                            {formatRecordingTime(recordingTime)}
                        </Text>
                    </View>
                ) : (
                    <View className="items-center">
                        <Text className="text-5xl">🎤</Text>
                        <Text className="text-white text-lg font-semibold mt-2">
                            Tap to Start
                        </Text>
                    </View>
                )}
            </TouchableOpacity>
        </View>
    );
}
