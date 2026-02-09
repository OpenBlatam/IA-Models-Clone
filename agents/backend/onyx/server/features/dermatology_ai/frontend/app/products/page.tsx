'use client';

import React, { useState } from 'react';
import { apiClient } from '@/lib/api/client';
import { ProductInfo, ProductSearchParams } from '@/lib/types/api';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { ShoppingBag } from 'lucide-react';
import { toastMessages, showInfo } from '@/lib/utils/toastUtils';
import { useMutation } from '@/lib/hooks/useMutation';
import { PageLayout } from '@/components/layout/PageLayout';
import { PageHeader } from '@/components/layout/PageHeader';
import { Select } from '@/components/ui/Select';
import { FormSection } from '@/components/ui/FormSection';
import { ProductCard } from '@/components/products/ProductCard';
import { EmptyState } from '@/lib/utils/emptyStates';
import { Grid } from '@/components/ui/Grid';
import { SearchBar } from '@/components/ui/SearchBar';

export default function ProductsPage() {
  const [products, setProducts] = useState<ProductInfo[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('');

  const categories = [
    'cleanser',
    'moisturizer',
    'serum',
    'sunscreen',
    'toner',
    'exfoliant',
    'mask',
    'eye_cream',
  ];

  const searchMutation = useMutation<ProductInfo[], ProductSearchParams>({
    mutationFn: async (params) => {
      if (!params.query.trim()) {
        throw new Error(toastMessages.enterSearchTerm);
      }
      const response = await apiClient.searchProducts(params);
      return response.products || [];
    },
    onSuccess: (data) => {
      setProducts(data);
      if (data.length === 0) {
        showInfo(toastMessages.noProductsFound);
      }
    },
    errorMessage: toastMessages.searchFailed,
  });

  const handleSearch = () => {
    const params: ProductSearchParams = {
      query: searchQuery,
      category: selectedCategory || undefined,
      limit: 20,
    };
    searchMutation.mutate(params);
  };

  return (
    <PageLayout>
      <PageHeader
        title="Products"
        description="Discover and compare skincare products"
        icon={ShoppingBag}
      />

        {/* Search Section */}
        <Card className="mb-8">
          <CardContent className="p-6">
            <FormSection spacing={4}>
              <div className="flex flex-col md:flex-row gap-4">
                <div className="flex-1">
                  <SearchBar
                    placeholder="Search by name, brand, or ingredient..."
                    onSearch={setSearchQuery}
                    className="w-full"
                  />
                </div>
                <div className="md:w-48">
                  <Select
                    value={selectedCategory}
                    onChange={(e) => setSelectedCategory(e.target.value)}
                    options={[
                      { value: '', label: 'All categories' },
                      ...categories.map((cat) => ({
                        value: cat,
                        label: cat.charAt(0).toUpperCase() + cat.slice(1),
                      })),
                    ]}
                    fullWidth
                  />
                </div>
                <Button onClick={handleSearch} isLoading={searchMutation.isLoading}>
                  Search
                </Button>
              </div>
            </FormSection>
          </CardContent>
        </Card>

        {/* Products Grid */}
        {searchMutation.isLoading ? (
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
            <p className="text-gray-600 dark:text-gray-400">Searching products...</p>
          </div>
        ) : products.length > 0 ? (
          <Grid cols={{ base: 1, md: 2, lg: 3 }} gap={6}>
            {products.map((product) => (
              <ProductCard key={product.product_id} product={product} />
            ))}
          </Grid>
        ) : (
          <EmptyState
            icon={<ShoppingBag className="h-16 w-16 text-gray-400" />}
            title="No products found"
            description="Try different search terms or browse all categories"
          />
        )}
      </div>
    </div>
  );
}

