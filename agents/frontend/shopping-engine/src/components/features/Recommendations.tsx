'use client';

import { motion } from 'framer-motion';
import { Star, ExternalLink, ArrowRight } from 'lucide-react';
import { Card, CardContent } from '@/src/components/ui/Card';
import { Button } from '@/src/components/ui/Button';
import type { Recommendation, RecommendationType } from '@/src/types/api';

interface RecommendationsProps {
    recommendations: Recommendation[];
    type: RecommendationType;
    onSelect?: (recommendation: Recommendation) => void;
}

const typeLabels: Record<RecommendationType, string> = {
    alternatives: 'Alternative Products',
    upgrades: 'Upgrade Options',
    accessories: 'Accessories',
    bundle: 'Bundle Suggestions',
};

const typeDescriptions: Record<RecommendationType, string> = {
    alternatives: 'Similar products that might interest you',
    upgrades: 'Premium options with better features',
    accessories: 'Compatible accessories and add-ons',
    bundle: 'Save more with these bundles',
};

interface RecommendationItemProps {
    rec: Recommendation;
    index: number;
    onSelect: (rec: Recommendation) => void;
    onKeyDownSelect: (e: React.KeyboardEvent, rec: Recommendation) => void;
}

const RecommendationItem = ({
    rec,
    index,
    onSelect,
    onKeyDownSelect,
}: RecommendationItemProps) => {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
        >
            <Card
                variant="bordered"
                isHoverable
                isClickable
                onClick={() => onSelect(rec)}
                onKeyDown={(e) => onKeyDownSelect(e, rec)}
                className="h-full flex flex-col"
            >
                {/* Image */}
                {rec.image_url && (
                    <div className="relative h-40 rounded-t-xl overflow-hidden bg-card">
                        <img
                            src={rec.image_url}
                            alt={rec.name}
                            className="w-full h-full object-cover"
                        />
                        {rec.similarity_score && (
                            <div className="absolute top-2 right-2 px-2 py-1 rounded-full bg-background/80 backdrop-blur-sm text-xs font-medium text-primary">
                                {Math.round(rec.similarity_score * 100)}% match
                            </div>
                        )}
                    </div>
                )}

                <CardContent className="flex-1 flex flex-col">
                    {/* Product Info */}
                    <div className="flex-1">
                        <div className="flex items-start justify-between gap-2">
                            <div>
                                <h3 className="font-semibold text-text line-clamp-2">{rec.name}</h3>
                                {rec.brand && (
                                    <p className="text-sm text-text-muted mt-1">{rec.brand}</p>
                                )}
                            </div>
                            {rec.rating && (
                                <div className="flex items-center gap-1 flex-shrink-0">
                                    <Star className="w-4 h-4 text-accent-warning fill-accent-warning" aria-hidden="true" />
                                    <span className="text-sm text-text-muted">{rec.rating.toFixed(1)}</span>
                                </div>
                            )}
                        </div>

                        <p className="text-sm text-text-muted mt-3 line-clamp-2">
                            {rec.reason}
                        </p>
                    </div>

                    {/* Bottom Section */}
                    <div className="flex items-center justify-between mt-4 pt-4 border-t border-white/10">
                        {rec.price && (
                            <p className="text-lg font-bold text-text">
                                {rec.currency || '$'}{rec.price.toLocaleString()}
                            </p>
                        )}
                        {rec.url ? (
                            <a
                                href={rec.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                onClick={(e) => e.stopPropagation()}
                                className="flex items-center gap-1 text-sm text-primary hover:text-primary-light transition-colors"
                                tabIndex={0}
                            >
                                View <ExternalLink className="w-3 h-3" aria-hidden="true" />
                            </a>
                        ) : (
                            <Button
                                variant="ghost"
                                size="sm"
                                rightIcon={<ArrowRight className="w-4 h-4" />}
                            >
                                Details
                            </Button>
                        )}
                    </div>
                </CardContent>
            </Card>
        </motion.div>
    );
};

export const Recommendations = ({
    recommendations,
    type,
    onSelect,
}: RecommendationsProps) => {
    const handleSelect = (rec: Recommendation) => {
        onSelect?.(rec);
    };

    const handleKeyDownSelect = (e: React.KeyboardEvent, rec: Recommendation) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            handleSelect(rec);
        }
    };

    if (recommendations.length === 0) {
        return (
            <Card variant="bordered" className="text-center py-12">
                <CardContent>
                    <p className="text-text-muted">No recommendations available</p>
                </CardContent>
            </Card>
        );
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h2 className="text-2xl font-bold text-text">{typeLabels[type]}</h2>
                <p className="text-text-muted mt-1">{typeDescriptions[type]}</p>
            </div>

            {/* Recommendations Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {recommendations.map((rec, index) => (
                    <RecommendationItem
                        key={`${rec.name}-${index}`}
                        rec={rec}
                        index={index}
                        onSelect={handleSelect}
                        onKeyDownSelect={handleKeyDownSelect}
                    />
                ))}
            </div>
        </div>
    );
};

export default Recommendations;
