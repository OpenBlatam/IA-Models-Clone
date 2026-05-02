'use client';

import { motion } from 'framer-motion';
import { Star, ArrowUpRight, Percent, TrendingDown, TrendingUp, Minus } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/src/components/ui/Card';
import type { VendorPrice } from '@/src/types/api';

interface PriceComparisonProps {
    prices: VendorPrice[];
    bestDeal?: VendorPrice;
    priceRange?: {
        min: number;
        max: number;
        average: number;
    };
}

const getTrendIcon = (price: number, average: number) => {
    const diff = ((price - average) / average) * 100;

    if (diff < -5) {
        return <TrendingDown className="w-4 h-4 text-accent-success" />;
    }
    if (diff > 5) {
        return <TrendingUp className="w-4 h-4 text-accent-error" />;
    }
    return <Minus className="w-4 h-4 text-text-muted" />;
};

interface PriceComparisonRowProps {
    priceItem: VendorPrice;
    index: number;
    isBestDeal: boolean;
    averagePrice?: number;
}

const PriceComparisonRow = ({
    priceItem,
    index,
    isBestDeal,
    averagePrice,
}: PriceComparisonRowProps) => {
    return (
        <motion.tr
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            className={`
                border-b border-white/5 hover:bg-card-hover transition-colors
                ${isBestDeal ? 'bg-accent-success/10' : ''}
            `}
        >
            <td className="py-4 px-4">
                <div className="flex items-center gap-2">
                    {isBestDeal && (
                        <Star className="w-4 h-4 text-accent-warning fill-accent-warning" aria-hidden="true" />
                    )}
                    <a
                        href={priceItem.vendor_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-medium text-text hover:text-primary transition-colors flex items-center gap-1"
                        tabIndex={0}
                    >
                        {priceItem.vendor}
                        <ArrowUpRight className="w-3 h-3" aria-hidden="true" />
                    </a>
                </div>
            </td>
            <td className="py-4 px-4 text-right">
                <div>
                    <span className="font-bold text-text">
                        {priceItem.currency} ${priceItem.price.toLocaleString()}
                    </span>
                    {priceItem.original_price && priceItem.original_price > priceItem.price && (
                        <span className="ml-2 text-sm text-text-muted line-through">
                            ${priceItem.original_price.toLocaleString()}
                        </span>
                    )}
                </div>
            </td>
            <td className="py-4 px-4 text-right">
                {priceItem.discount_percentage ? (
                    <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-accent-success/20 text-accent-success text-sm font-medium">
                        <Percent className="w-3 h-3" aria-hidden="true" />
                        {priceItem.discount_percentage}% OFF
                    </span>
                ) : (
                    <span className="text-text-muted">—</span>
                )}
            </td>
            <td className="py-4 px-4 text-center">
                {averagePrice !== undefined && getTrendIcon(priceItem.price, averagePrice)}
            </td>
            <td className="py-4 px-4 text-center">
                <span className={`
                    text-sm font-medium
                    ${priceItem.availability === 'In Stock' || priceItem.availability === 'in_stock'
                        ? 'text-accent-success'
                        : priceItem.availability === 'Limited' || priceItem.availability === 'limited'
                            ? 'text-accent-warning'
                            : 'text-accent-error'
                    }
                `}>
                    {priceItem.availability}
                </span>
            </td>
        </motion.tr>
    );
};

export const PriceComparison = ({
    prices,
    bestDeal,
    priceRange,
}: PriceComparisonProps) => {
    if (prices.length === 0) {
        return (
            <Card variant="bordered" className="text-center py-12">
                <CardContent>
                    <p className="text-text-muted">No prices to compare</p>
                </CardContent>
            </Card>
        );
    }

    return (
        <div className="space-y-6">
            {/* Price Range Summary */}
            {priceRange && (
                <Card variant="glass">
                    <CardContent>
                        <div className="grid grid-cols-3 gap-4 text-center">
                            <div>
                                <p className="text-sm text-text-muted">Lowest</p>
                                <p className="text-xl font-bold text-accent-success">
                                    ${priceRange.min.toLocaleString()}
                                </p>
                            </div>
                            <div>
                                <p className="text-sm text-text-muted">Average</p>
                                <p className="text-xl font-bold text-text">
                                    ${priceRange.average.toLocaleString()}
                                </p>
                            </div>
                            <div>
                                <p className="text-sm text-text-muted">Highest</p>
                                <p className="text-xl font-bold text-accent-error">
                                    ${priceRange.max.toLocaleString()}
                                </p>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* Comparison Table */}
            <Card variant="bordered">
                <CardHeader>
                    <CardTitle>Price Comparison</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="border-b border-white/10">
                                    <th className="text-left py-3 px-4 text-sm font-semibold text-text-muted">Vendor</th>
                                    <th className="text-right py-3 px-4 text-sm font-semibold text-text-muted">Price</th>
                                    <th className="text-right py-3 px-4 text-sm font-semibold text-text-muted">Discount</th>
                                    <th className="text-center py-3 px-4 text-sm font-semibold text-text-muted">Trend</th>
                                    <th className="text-center py-3 px-4 text-sm font-semibold text-text-muted">Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {prices.map((priceItem, index) => (
                                    <PriceComparisonRow
                                        key={`${priceItem.vendor}-${index}`}
                                        priceItem={priceItem}
                                        index={index}
                                        isBestDeal={!!(bestDeal && priceItem.vendor === bestDeal.vendor)}
                                        averagePrice={priceRange?.average}
                                    />
                                ))}
                            </tbody>
                        </table>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
};

export default PriceComparison;
