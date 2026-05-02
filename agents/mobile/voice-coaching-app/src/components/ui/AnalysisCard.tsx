import React from "react";
import { View, Text, TouchableOpacity } from "react-native";
/// <reference types="nativewind/types" />
import { Card } from "./Card";
import { VoiceAnalysis } from "../../types";
import { formatDate, formatDuration, getScoreTextColor } from "../../utils";

interface AnalysisCardProps {
    analysis: VoiceAnalysis;
    onPress?: () => void;
}

/**
 * Card displaying a voice analysis summary
 */
export function AnalysisCard({ analysis, onPress }: AnalysisCardProps) {
    const scoreColor = getScoreTextColor(analysis.score);

    const content = (
        <Card variant="glass" className="p-4">
            <View className="flex-row items-center justify-between">
                <View className="flex-1">
                    <Text className="text-white font-semibold text-lg">
                        {formatDate(analysis.timestamp)}
                    </Text>
                    <View className="flex-row items-center mt-1 space-x-4">
                        <Text className="text-text-secondary">
                            ⏱️ {formatDuration(analysis.duration)}
                        </Text>
                        <Text className="text-text-secondary">
                            💬 {analysis.feedback.length} tips
                        </Text>
                    </View>
                </View>
                <View className="items-end">
                    <Text className="text-3xl font-bold" style={{ color: scoreColor.includes('text-') ? undefined : scoreColor }}>
                        {analysis.score}
                    </Text>
                    <Text className="text-text-muted text-xs">Score</Text>
                </View>
            </View>
        </Card>
    );

    if (onPress) {
        return (
            <TouchableOpacity onPress={onPress} activeOpacity={0.8}>
                {content}
            </TouchableOpacity>
        );
    }

    return content;
}
