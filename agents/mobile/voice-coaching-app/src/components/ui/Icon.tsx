import React from "react";
import { Ionicons, MaterialCommunityIcons, FontAwesome5 } from "@expo/vector-icons";
import { COLORS } from "../../constants";

type IconSet = "ionicons" | "material" | "fontawesome";

interface IconProps {
    name: string;
    size?: number;
    color?: string;
    set?: IconSet;
}

/**
 * Unified Icon component supporting multiple icon libraries
 */
export function Icon({
    name,
    size = 24,
    color = COLORS.text.primary,
    set = "ionicons",
}: IconProps) {
    switch (set) {
        case "material":
            return <MaterialCommunityIcons name={name as keyof typeof MaterialCommunityIcons.glyphMap} size={size} color={color} />;
        case "fontawesome":
            return <FontAwesome5 name={name as keyof typeof FontAwesome5.glyphMap} size={size} color={color} />;
        case "ionicons":
        default:
            return <Ionicons name={name as keyof typeof Ionicons.glyphMap} size={size} color={color} />;
    }
}

// Common icon presets
export const Icons = {
    home: (props?: Partial<IconProps>) => <Icon name="home" {...props} />,
    history: (props?: Partial<IconProps>) => <Icon name="time" {...props} />,
    settings: (props?: Partial<IconProps>) => <Icon name="settings" {...props} />,
    mic: (props?: Partial<IconProps>) => <Icon name="mic" {...props} />,
    micOff: (props?: Partial<IconProps>) => <Icon name="mic-off" {...props} />,
    play: (props?: Partial<IconProps>) => <Icon name="play" {...props} />,
    stop: (props?: Partial<IconProps>) => <Icon name="stop" {...props} />,
    pause: (props?: Partial<IconProps>) => <Icon name="pause" {...props} />,
    check: (props?: Partial<IconProps>) => <Icon name="checkmark-circle" {...props} />,
    close: (props?: Partial<IconProps>) => <Icon name="close" {...props} />,
    back: (props?: Partial<IconProps>) => <Icon name="arrow-back" {...props} />,
    forward: (props?: Partial<IconProps>) => <Icon name="arrow-forward" {...props} />,
    chevronRight: (props?: Partial<IconProps>) => <Icon name="chevron-forward" {...props} />,
    star: (props?: Partial<IconProps>) => <Icon name="star" {...props} />,
    tips: (props?: Partial<IconProps>) => <Icon name="bulb" {...props} />,
    user: (props?: Partial<IconProps>) => <Icon name="person" {...props} />,
    analytics: (props?: Partial<IconProps>) => <Icon name="stats-chart" {...props} />,
} as const;
