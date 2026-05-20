'use client';

import React from 'react';
import { ImageUpload } from '@/components/analysis/ImageUpload';
import { AnalysisResults } from '@/components/analysis/AnalysisResults';
import { RecommendationsDisplay } from '@/components/recommendations/RecommendationsDisplay';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Tabs } from '@/components/ui/Tabs';
import { Badge } from '@/components/ui/Badge';
import { Alert } from '@/components/ui/Alert';
import { apiClient } from '@/lib/api/client';
import { AnalysisResponse, RecommendationsResponse } from '@/lib/types/api';
import { Sparkles, Upload, FileImage, Video, Zap, TrendingUp, Shield, CheckCircle2, Star, ArrowRight, Play, BarChart3, Wand2 } from 'lucide-react';
import { useAuth } from '@/lib/contexts/AuthContext';
import { useScroll } from '@/lib/hooks/useScroll';
import { useMousePosition } from '@/lib/hooks/useMousePosition';
import { useMutation } from '@/lib/hooks/useMutation';
import { FloatingParticles } from '@/components/effects/FloatingParticles';
import { AnimatedWaves } from '@/components/effects/AnimatedWaves';
import { GradientRings } from '@/components/effects/GradientRings';
import { FeatureCard } from '@/components/ui/FeatureCard';
import { Grid } from '@/components/ui/Grid';
import { InfoCard } from '@/components/ui/InfoCard';
import { StepCard } from '@/components/ui/StepCard';
import { LoadingOverlay } from '@/components/ui/LoadingOverlay';
import { SectionHeader } from '@/components/ui/SectionHeader';
import { TabButton } from '@/components/ui/TabButton';
import { ButtonGroup } from '@/components/ui/ButtonGroup';
import { Divider } from '@/components/ui/Divider';
import { useFileUpload } from '@/lib/hooks/useFileUpload';
import { ResultCard } from '@/components/ui/ResultCard';

export default function HomePage() {
  const { isAuthenticated } = useAuth();
  const scrollY = useScroll();
  const mousePosition = useMousePosition();

  const {
    selectedFile,
    setSelectedFile,
    analysis,
    setAnalysis,
    recommendations,
    setRecommendations,
    activeTab,
    setActiveTab,
    uploadType,
    setUploadType,
  } = useFileUpload();

  const analyzeMutation = useMutation<AnalysisResponse, File>({
    mutationFn: async (file: File) => {
      if (!file) throw new Error('Please select an image first');
      return await apiClient.analyzeImage(file, true);
    },
    onSuccess: (result) => {
      setAnalysis(result);
    },
    successMessage: 'Analysis complete!',
    errorMessage: 'Analysis failed. Please try again.',
  });

  const recommendationsMutation = useMutation<RecommendationsResponse, File>({
    mutationFn: async (file: File) => {
      if (!file) throw new Error('Please select an image first');
      return await apiClient.getRecommendations(file, true);
    },
    onSuccess: (result) => {
      setRecommendations(result);
    },
    successMessage: 'Recommendations ready!',
    errorMessage: 'Failed to generate recommendations. Please try again.',
  });

  const handleAnalyze = () => {
    if (!selectedFile) return;
    analyzeMutation.mutate(selectedFile);
  };

  const handleGetRecommendations = () => {
    if (!selectedFile) return;
    recommendationsMutation.mutate(selectedFile);
  };

  return (
    <div className="min-h-screen bg-white dark:bg-gray-950 relative">
      {/* Custom cursor effect - subtle glow */}
      <div 
        className="fixed w-8 h-8 rounded-full bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-pink-500/10 dark:from-blue-400/10 dark:via-purple-400/10 dark:to-pink-400/10 pointer-events-none z-50 blur-sm transition-all duration-200 ease-out"
        style={{
          left: `${mousePosition.x - 16}px`,
          top: `${mousePosition.y - 16}px`,
          transform: 'translate(-50%, -50%)',
          opacity: mousePosition.x === 0 && mousePosition.y === 0 ? 0 : 0.6,
        }}
      />
      
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        {/* Background Gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-950 dark:to-gray-900" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(59,130,246,0.05),transparent_50%)] dark:bg-[radial-gradient(circle_at_50%_50%,rgba(59,130,246,0.1),transparent_50%)]" />
        
        {/* Animated gradient orbs */}
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-400/20 dark:bg-blue-500/10 rounded-full blur-3xl animate-float" />
        <div className="absolute top-1/4 right-1/4 w-96 h-96 bg-purple-400/20 dark:bg-purple-500/10 rounded-full blur-3xl animate-float" style={{ animationDelay: '1s' }} />
        <div className="absolute bottom-0 left-1/2 w-96 h-96 bg-pink-400/20 dark:bg-pink-500/10 rounded-full blur-3xl animate-float" style={{ animationDelay: '2s' }} />
        <div className="absolute top-1/2 left-0 w-64 h-64 bg-cyan-400/15 dark:bg-cyan-500/8 rounded-full blur-3xl animate-float" style={{ animationDelay: '0.5s' }} />
        <div className="absolute bottom-1/4 right-0 w-80 h-80 bg-indigo-400/15 dark:bg-indigo-500/8 rounded-full blur-3xl animate-float" style={{ animationDelay: '1.5s' }} />
        
        {/* Grid pattern overlay */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px] opacity-40 dark:opacity-20" />
        
        {/* Shimmer effect */}
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full animate-shimmer-slow dark:via-white/5" />
        
        {/* Animated wave effects */}
        <AnimatedWaves height={128} />
        
        {/* Radial gradient rings */}
        <GradientRings count={2} sizes={[800, 600]} />
        
        {/* Floating particles */}
        <FloatingParticles count={30} />
        
        {/* Animated lines */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none opacity-20 dark:opacity-10">
          <svg className="absolute inset-0 w-full h-full" style={{ transform: `translateY(${scrollY * 0.1}px)` }}>
            <defs>
              <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="rgba(59, 130, 246, 0.3)" />
                <stop offset="50%" stopColor="rgba(147, 51, 234, 0.3)" />
                <stop offset="100%" stopColor="rgba(236, 72, 153, 0.3)" />
              </linearGradient>
            </defs>
            <path
              d="M 0,200 Q 400,100 800,200 T 1600,200"
              stroke="url(#lineGradient)"
              strokeWidth="2"
              fill="none"
              className="animate-draw-line"
            />
            <path
              d="M 0,400 Q 400,300 800,400 T 1600,400"
              stroke="url(#lineGradient)"
              strokeWidth="2"
              fill="none"
              className="animate-draw-line"
              style={{ animationDelay: '1s' }}
            />
          </svg>
        </div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16 lg:pt-32 lg:pb-24">
          {/* Hero Content */}
          <div className="text-center mb-12 animate-fade-in-up">
            <div className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/50 dark:to-purple-950/50 border border-blue-200/50 dark:border-blue-900/50 mb-8 shadow-sm hover:shadow-lg hover:scale-105 transition-all duration-300 relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-400/0 via-blue-400/20 to-blue-400/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
              <div className="absolute inset-0 bg-gradient-to-br from-blue-500/0 to-purple-500/0 group-hover:from-blue-500/10 group-hover:to-purple-500/10 transition-all duration-500 rounded-full" />
              <Sparkles className="h-4 w-4 text-blue-600 dark:text-blue-400 animate-pulse relative z-10 group-hover:animate-spin-slow" />
              <span className="text-sm font-semibold text-blue-700 dark:text-blue-300 relative z-10">
                Powered by AI
              </span>
              {/* Sparkle particles */}
              <div className="absolute -top-1 -right-1 w-2 h-2 bg-blue-400 rounded-full opacity-0 group-hover:opacity-100 group-hover:animate-sparkle" />
              <div className="absolute -bottom-1 -left-1 w-1.5 h-1.5 bg-purple-400 rounded-full opacity-0 group-hover:opacity-100 group-hover:animate-sparkle" style={{ animationDelay: '0.3s' }} />
            </div>
            
            <h1 
              className="text-5xl md:text-6xl lg:text-7xl font-bold text-gray-900 dark:text-white mb-6 tracking-tight leading-tight relative"
              style={{ transform: `translateY(${scrollY * 0.05}px)` }}
            >
              <span className="relative z-10 inline-block hover:scale-105 transition-transform duration-300 group">
                <span className="inline-block hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-300">Analyze</span>
              </span>
              <span className="block bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent animate-gradient relative z-10 hover:scale-105 transition-transform duration-300 inline-block group">
                your skin
                <span className="absolute inset-0 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 blur-2xl opacity-30 -z-10 group-hover:opacity-50 transition-opacity duration-300" />
                {/* Animated underline */}
                <span className="absolute bottom-0 left-0 w-0 h-1 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 group-hover:w-full transition-all duration-500 rounded-full" />
              </span>
            </h1>
            
            <p className="text-xl md:text-2xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto mb-8 leading-relaxed font-light relative">
              <span className="inline-block hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-300">Upload</span>
              <span className="inline-block mx-1">a</span>
              <span className="inline-block hover:text-purple-600 dark:hover:text-purple-400 transition-colors duration-300">photo.</span>
              <span className="inline-block mx-1">Get</span>
              <span className="inline-block hover:text-pink-600 dark:hover:text-pink-400 transition-colors duration-300">insights.</span>
            </p>
            
            {/* Usage stats - Suno style */}
            <div className="flex items-center justify-center gap-8 mt-6 mb-8 text-sm text-gray-500 dark:text-gray-400">
              <div className="flex items-center gap-2">
                <span className="text-2xl font-bold text-gray-900 dark:text-white">10K+</span>
                <span>analyses</span>
              </div>
              <div className="w-1 h-1 rounded-full bg-gray-400" />
              <div className="flex items-center gap-2">
                <span className="text-2xl font-bold text-gray-900 dark:text-white">2s</span>
                <span>avg</span>
              </div>
              <div className="w-1 h-1 rounded-full bg-gray-400" />
              <div className="flex items-center gap-2">
                <span className="text-2xl font-bold text-gray-900 dark:text-white">99%</span>
                <span>accurate</span>
              </div>
            </div>

            {/* Quick features - Suno style */}
            <div className="flex flex-wrap items-center justify-center gap-4 mt-8 mb-8 animate-slide-up">
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/60 dark:bg-gray-900/60 backdrop-blur-sm border border-gray-200/50 dark:border-gray-800/50 hover:scale-105 hover:shadow-lg transition-all duration-300 group text-xs">
                <CheckCircle2 className="h-4 w-4 text-green-500 group-hover:scale-110 transition-transform duration-300" />
                <span className="font-medium text-gray-700 dark:text-gray-300">No signup required</span>
              </div>
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/60 dark:bg-gray-900/60 backdrop-blur-sm border border-gray-200/50 dark:border-gray-800/50 hover:scale-105 hover:shadow-lg transition-all duration-300 group text-xs">
                <Star className="h-4 w-4 text-yellow-500 fill-yellow-500 group-hover:rotate-12 transition-transform duration-300" />
                <span className="font-medium text-gray-700 dark:text-gray-300">Free forever</span>
              </div>
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/60 dark:bg-gray-900/60 backdrop-blur-sm border border-gray-200/50 dark:border-gray-800/50 hover:scale-105 hover:shadow-lg transition-all duration-300 group text-xs">
                <BarChart3 className="h-4 w-4 text-blue-500 group-hover:scale-110 transition-transform duration-300" />
                <span className="font-medium text-gray-700 dark:text-gray-300">Instant analysis</span>
              </div>
            </div>

            {!isAuthenticated && (
              <div className="flex items-center justify-center gap-2 mb-8">
                <Badge variant="info" size="sm" className="bg-blue-50 dark:bg-blue-950/50 border-blue-200 dark:border-blue-900/50">
                  <Sparkles className="h-3 w-3 mr-1" />
                  <span className="text-xs">Sign in to save history</span>
                </Badge>
              </div>
            )}
          </div>

          {/* Main Upload Area */}
          <div 
            data-upload-area
            className="max-w-4xl mx-auto animate-fade-in-up" 
            style={{ animationDelay: '0.2s', transform: `translateY(${scrollY * 0.03}px)` }}
          >
            <Card className="border-0 shadow-2xl bg-white/90 dark:bg-gray-900/90 backdrop-blur-md hover:shadow-3xl transition-all duration-500 relative overflow-hidden group">
              {/* Animated border gradient */}
              <div className="absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500">
                <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-pink-500/20 blur-xl" />
                <div className="absolute inset-[1px] rounded-xl bg-white dark:bg-gray-900" />
              </div>
              
              {/* Glowing corners */}
              <div className="absolute top-0 left-0 w-32 h-32 bg-blue-500/10 rounded-full blur-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 animate-pulse-glow" />
              <div className="absolute bottom-0 right-0 w-32 h-32 bg-purple-500/10 rounded-full blur-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 animate-pulse-glow" style={{ animationDelay: '0.5s' }} />
              
              {/* Animated border ring */}
              <div className="absolute inset-0 rounded-xl border-2 border-transparent bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 opacity-0 group-hover:opacity-20 transition-opacity duration-500 -z-10 blur-sm" />
              <div className="absolute inset-[2px] rounded-xl bg-white dark:bg-gray-900" />
              <div className="relative z-10 p-6 lg:p-8">
                <Tabs
                  tabs={[
                    {
                      id: 'image',
                      label: 'Image',
                      icon: <FileImage className="h-4 w-4" />,
                      content: (
                        <div className="mt-6">
                          <ImageUpload
                            onFileSelect={setSelectedFile}
                            currentFile={selectedFile}
                            accept="image/*"
                          />
                        </div>
                      ),
                    },
                    {
                      id: 'video',
                      label: 'Video',
                      icon: <Video className="h-4 w-4" />,
                      content: (
                        <div className="mt-6">
                          <ImageUpload
                            onFileSelect={setSelectedFile}
                            currentFile={selectedFile}
                            accept="video/*"
                          />
                          <Alert variant="info" className="mt-4 text-sm">
                            Videos analyzed via key frame extraction.
                          </Alert>
                        </div>
                      ),
                    },
                  ]}
                  onTabChange={(tabId) => {
                    setUploadType(tabId as 'image' | 'video');
                    setSelectedFile(null);
                    setAnalysis(null);
                    setRecommendations(null);
                  }}
                />
              </div>

              {selectedFile && (
                <div className="relative z-10 px-6 pb-6 border-t border-gray-200 dark:border-gray-800 pt-6 animate-fade-in">
                  <LoadingOverlay
                    isLoading={analyzeMutation.isLoading || recommendationsMutation.isLoading}
                    message={
                      analyzeMutation.isLoading
                        ? 'Analyzing...'
                        : recommendationsMutation.isLoading
                        ? 'Generating...'
                        : 'Processing...'
                    }
                    subMessage="Just a moment"
                  />
                  
                  <div className="flex flex-col sm:flex-row gap-3">
                    <Button
                      onClick={uploadType === 'image' ? handleAnalyze : () => {
                        if (!selectedFile) return;
                        analyzeMutation.mutate(selectedFile);
                      }}
                      isLoading={analyzeMutation.isLoading}
                      className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 shadow-lg hover:shadow-2xl hover:scale-[1.02] transition-all duration-300 relative overflow-hidden group"
                      size="lg"
                    >
                      <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
                      <div className="absolute inset-0 bg-gradient-to-br from-blue-400/0 via-purple-400/0 to-pink-400/0 group-hover:from-blue-400/20 group-hover:via-purple-400/20 group-hover:to-pink-400/20 transition-all duration-500" />
                      <Upload className="h-4 w-4 mr-2 relative z-10 group-hover:rotate-12 transition-transform duration-300" />
                      <span className="relative z-10 font-semibold">{uploadType === 'image' ? 'Analyze' : 'Analyze video'}</span>
                    </Button>
                    {uploadType === 'image' && (
                      <Button
                        onClick={handleGetRecommendations}
                        isLoading={recommendationsMutation.isLoading}
                        variant="secondary"
                        className="flex-1 border-2 border-purple-200 dark:border-purple-800 hover:border-purple-300 dark:hover:border-purple-700 hover:bg-purple-50 dark:hover:bg-purple-950/30 hover:scale-[1.02] transition-all duration-300 relative overflow-hidden group"
                        size="lg"
                      >
                        <div className="absolute inset-0 bg-gradient-to-r from-purple-100/0 via-purple-100/30 to-purple-100/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700 dark:from-purple-900/0 dark:via-purple-900/30 dark:to-purple-900/0" />
                        <div className="absolute inset-0 bg-gradient-to-br from-purple-400/0 to-pink-400/0 group-hover:from-purple-400/10 group-hover:to-pink-400/10 transition-all duration-500" />
                        <Sparkles className="h-4 w-4 mr-2 relative z-10 group-hover:rotate-180 transition-transform duration-500" />
                        <span className="relative z-10 font-semibold">Get recommendations</span>
                      </Button>
                    )}
                  </div>
                </div>
              )}
            </Card>
          </div>

          {!isAuthenticated && (
            <div className="mt-8 max-w-2xl mx-auto animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
              <Alert variant="info" className="bg-blue-50/50 dark:bg-blue-950/20 border-blue-200 dark:border-blue-900/50 backdrop-blur-sm hover:shadow-lg transition-shadow duration-300 text-sm">
                <strong>Tip:</strong> Sign up to save your history.
              </Alert>
            </div>
          )}

          {/* Features Grid */}
          {!analysis && !recommendations && (
            <Grid
              cols={{ base: 1, md: 3 }}
              gap={6}
              maxWidth="5xl"
              className="mt-20 animate-fade-in-up"
              style={{ animationDelay: '0.4s', transform: `translateY(${scrollY * 0.02}px)` }}
            >
              <FeatureCard
                icon={Zap}
                title="Lightning Fast"
                description="Instant results"
                iconColor="text-blue-600 dark:text-blue-400"
              />
              <FeatureCard
                icon={TrendingUp}
                title="Smart Recommendations"
                description="Tailored for you"
                iconColor="text-purple-600 dark:text-purple-400"
              />
              <FeatureCard
                icon={Shield}
                title="Privacy"
                description="Fully secure"
                iconColor="text-green-600 dark:text-green-400"
              />
            </Grid>
          )}
        </div>
      </div>

      {/* Results Section */}
      {(analysis || recommendations) && (
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          {/* Background decoration */}
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-blue-50/30 to-transparent dark:via-blue-950/20" />
          
          {/* Animated background elements */}
          <div className="absolute top-10 left-10 w-32 h-32 bg-blue-400/10 dark:bg-blue-500/5 rounded-full blur-2xl animate-float" />
          <div className="absolute bottom-10 right-10 w-40 h-40 bg-purple-400/10 dark:bg-purple-500/5 rounded-full blur-2xl animate-float" style={{ animationDelay: '1s' }} />
          
          <div className="relative z-10 mb-8">
            <div className="flex space-x-1 border-b border-gray-200 dark:border-gray-800 bg-gray-50/80 dark:bg-gray-900/80 backdrop-blur-sm rounded-t-xl p-1.5 shadow-inner">
              <TabButton
                label="Analysis"
                isActive={activeTab === 'analysis'}
                onClick={() => setActiveTab('analysis')}
              />
              {recommendations && (
                <TabButton
                  label="Recommendations"
                  isActive={activeTab === 'recommendations'}
                  onClick={() => setActiveTab('recommendations')}
                />
              )}
            </div>
          </div>

          <div className="relative z-10 animate-fade-in-up">
            {activeTab === 'analysis' && analysis && (
              <ResultCard variant="analysis">
                <AnalysisResults
                  analysis={analysis.analysis}
                  analysisId={analysis.analysis.analysis_id}
                />
              </ResultCard>
            )}

            {activeTab === 'recommendations' && recommendations && (
              <ResultCard variant="recommendations">
                <RecommendationsDisplay
                  recommendations={recommendations.recommendations}
                />
              </ResultCard>
            )}
          </div>
        </div>
      )}

      {/* How it works - Suno style */}
      {!analysis && !recommendations && (
        <div className="relative max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <SectionHeader
            title="How it works"
            description="Simple. Fast. Free forever."
            align="center"
            size="md"
          />

          <Grid cols={{ base: 1, md: 3 }} gap={8}>
            <StepCard
              step={1}
              title="Upload"
              description="Drop your photo"
              iconColor="text-blue-600 dark:text-blue-400"
            />
            <StepCard
              step={2}
              title="Analyze"
              description="AI analyzes"
              iconColor="text-purple-600 dark:text-purple-400"
            />
            <StepCard
              step={3}
              title="Done"
              description="Get insights"
              iconColor="text-pink-600 dark:text-pink-400"
            />
          </Grid>
        </div>
      )}

      {/* FAQ or Benefits Section - Suno style */}
      {!analysis && !recommendations && (
        <div className="relative max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <Divider spacing="lg" />
          <SectionHeader
            title="Why us?"
            align="center"
            size="md"
          />

          <Grid cols={{ base: 1, md: 2 }} gap={6} maxWidth="4xl">
            <InfoCard
              title="No credit card"
              description="Start free. Always free."
              hoverColor="blue"
            />
            <InfoCard
              title="Unlimited"
              description="Unlimited analyses. No limits."
              hoverColor="purple"
            />
            <InfoCard
              title="Advanced AI"
              description="State-of-the-art AI."
              hoverColor="pink"
            />
            <InfoCard
              title="Always improving"
              description="Constantly improving."
              hoverColor="green"
            />
          </Grid>
        </div>
      )}

      {/* CTA Section - Suno style */}
      {!analysis && !recommendations && !isAuthenticated && (
        <div className="relative max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center p-12 rounded-2xl bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950/50 dark:to-purple-950/50 border border-blue-200/50 dark:border-blue-900/50">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Ready to get started?
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-400 mb-8 max-w-2xl mx-auto">
              Upload your photo above and see the magic happen.
            </p>
            <div className="flex items-center justify-center gap-4">
              <Button 
                size="lg" 
                className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 shadow-lg hover:shadow-xl transition-all"
                onClick={() => {
                  document.querySelector('[data-upload-area]')?.scrollIntoView({ behavior: 'smooth' });
                }}
              >
                Get Started
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

