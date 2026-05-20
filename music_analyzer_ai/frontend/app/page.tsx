/**
 * Home page component - Suno.com clone.
 * Landing page with music creation interface, featured songs, and pricing.
 */

'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Button, Input } from '@/components/ui';
import {
  Play,
  Heart,
  Sparkles,
  Wand2,
  Shuffle,
  Check,
  Download,
  Twitter,
  Instagram,
  Music,
} from 'lucide-react';
import { ROUTES } from '@/lib/constants';

/**
 * Featured song interface.
 */
interface FeaturedSong {
  id: string;
  title: string;
  artist: string;
  image: string;
  plays: string;
  likes: string;
}

/**
 * Press logo interface.
 */
interface PressLogo {
  name: string;
  image: string;
}

/**
 * Feature interface.
 */
interface Feature {
  title: string;
  description: string;
}

/**
 * Pricing plan interface.
 */
interface PricingPlan {
  name: string;
  price: string;
  period: string;
  description: string;
  badge?: string;
  features: string[];
  buttonText: string;
  popular?: boolean;
}

/**
 * Home page component.
 * Provides Suno.com-style landing page with music creation interface.
 *
 * @returns Home page component
 */
export default function HomePage() {
  const [prompt, setPrompt] = useState('');
  const [billingPeriod, setBillingPeriod] = useState<'monthly' | 'yearly'>(
    'yearly'
  );

  // Featured songs data (top carousel) - dynamic, updates frequently on Suno
  const featuredSongs: FeaturedSong[] = [
    {
      id: '1',
      title: '𝕃𝕚𝕓𝕖𝕣𝕥𝕪 𝕠𝕣 ℂ𝕣𝕦𝕖𝕝𝕥𝕪...?',
      artist: '𝔻𝕚𝕞 ℂ𝕒𝕟𝕕𝕝𝕖𝕤',
      image: '/api/placeholder/300/300',
      plays: '85K',
      likes: '2.1K',
    },
    {
      id: '2',
      title: 'Play the Cards',
      artist: 'BettyByte',
      image: '/api/placeholder/300/300',
      plays: '6.8K',
      likes: '464',
    },
  ];

  // All songs for grid (in exact order from Suno snapshot)
  const allSongs: FeaturedSong[] = [
    {
      id: '1',
      title: '𝕃𝕚𝕓𝕖𝕣𝕥𝕪 𝕠𝕣 ℂ𝕣𝕦𝕖𝕝𝕥𝕪...?',
      artist: '𝔻𝕚𝕞 ℂ𝕒𝕟𝕕𝕝𝕖𝕤',
      image: '/api/placeholder/300/300',
      plays: '85K',
      likes: '2.1K',
    },
    {
      id: '2',
      title: 'Play the Cards',
      artist: 'BettyByte',
      image: '/api/placeholder/300/300',
      plays: '6.8K',
      likes: '465',
    },
    {
      id: '3',
      title: 'Memory',
      artist: 'JoonOc',
      image: '/api/placeholder/300/300',
      plays: '142K',
      likes: '4.6K',
    },
    {
      id: '4',
      title: 'Delapidated (Cover)',
      artist: 'DeeBeetttZZ',
      image: '/api/placeholder/300/300',
      plays: '82K',
      likes: '1.5K',
    },
    {
      id: '5',
      title: 'Parental Burnout',
      artist: 'The Beat Bastards',
      image: '/api/placeholder/300/300',
      plays: '30K',
      likes: '733',
    },
    {
      id: '6',
      title: 'take me with her',
      artist: 'Sol',
      image: '/api/placeholder/300/300',
      plays: '12K',
      likes: '256',
    },
    {
      id: '7',
      title: "Everything's Fine",
      artist: 'Staabsworth',
      image: '/api/placeholder/300/300',
      plays: '108K',
      likes: '3.7K',
    },
    {
      id: '8',
      title: 'Bittersweet Gravity',
      artist: 'Wistaria Addict',
      image: '/api/placeholder/300/300',
      plays: '72K',
      likes: '2.2K',
    },
    {
      id: '9',
      title: 'Mojave',
      artist: 'HighSpeedOutro9488',
      image: '/api/placeholder/300/300',
      plays: '35K',
      likes: '757',
    },
    {
      id: '10',
      title: "April's Gone",
      artist: 'Thomas Otten',
      image: '/api/placeholder/300/300',
      plays: '88K',
      likes: '2.4K',
    },
    {
      id: '11',
      title: 'Lo-Fi Rocks',
      artist: 'Stray Cat/Game Dev',
      image: '/api/placeholder/300/300',
      plays: '30K',
      likes: '498',
    },
    {
      id: '12',
      title: 'Something is going down',
      artist: 'XAVI',
      image: '/api/placeholder/300/300',
      plays: '58K',
      likes: '1.4K',
    },
    {
      id: '13',
      title: 'Signal Fade',
      artist: 'MakMuzical',
      image: '/api/placeholder/300/300',
      plays: '52K',
      likes: '1.0K',
    },
    {
      id: '14',
      title: "Moth Among the Stars (Live)",
      artist: 'Sam',
      image: '/api/placeholder/300/300',
      plays: '79K',
      likes: '1.9K',
    },
    {
      id: '15',
      title: 'passing by',
      artist: 'bleeder',
      image: '/api/placeholder/300/300',
      plays: '17K',
      likes: '638',
    },
    {
      id: '16',
      title: 'changes',
      artist: 'GTZY',
      image: '/api/placeholder/300/300',
      plays: '27K',
      likes: '713',
    },
    {
      id: '17',
      title: 'almost home',
      artist: 'Making Music Is Life',
      image: '/api/placeholder/300/300',
      plays: '49K',
      likes: '1.4K',
    },
    {
      id: '18',
      title: 'Muse',
      artist: 'sonoa',
      image: '/api/placeholder/300/300',
      plays: '14K',
      likes: '191',
    },
    {
      id: '19',
      title: 'lilies in the pond.',
      artist: 'a.astronaut',
      image: '/api/placeholder/300/300',
      plays: '33K',
      likes: '1.5K',
    },
    {
      id: '20',
      title: 'I am in - I am out EDM',
      artist: 'EUMAK4U',
      image: '/api/placeholder/300/300',
      plays: '38K',
      likes: '960',
    },
    {
      id: '21',
      title: 'Stuck In My Head',
      artist: 'Kostaboda',
      image: '/api/placeholder/300/300',
      plays: '39K',
      likes: '823',
    },
  ];

  // Press logos
  const pressLogos: PressLogo[] = [
    { name: 'Billboard', image: '/api/placeholder/120/40' },
    { name: 'Complex', image: '/api/placeholder/120/40' },
    { name: 'Forbes', image: '/api/placeholder/120/40' },
    { name: 'Rolling Stone', image: '/api/placeholder/120/40' },
    { name: 'Variety', image: '/api/placeholder/120/40' },
    { name: 'Wired', image: '/api/placeholder/120/40' },
  ];

  // Features
  const features: Feature[] = [
    {
      title: '10 Free songs, daily',
      description:
        'Turn any moment into customized music instantly — from your commute to inside jokes. Express what words can\'t. Free forever, no subscription needed.',
    },
    {
      title: 'Free AI music generator',
      description:
        "Discover what's possible when anyone can make music. Access the market-leading AI song generator to explore millions of songs—remixes, jokes, and raw emotion.",
    },
    {
      title: 'Share it with the world',
      description:
        "Make music that matters to you, then share it with people who'll feel it too. From your inner circle to millions of music fans, your next track can go far.",
    },
    {
      title: 'Experience the modern song maker',
      description:
        'Start, edit, remix—your way. Upload or record your own audio, rewrite lyrics, reorder sections, and reimagine your sound with powerful creative tools.',
    },
    {
      title: 'Create everyday. Keep it all.',
      description:
        'Make up to 500 custom songs a month, with full commercial rights on the Pro plan. Get inspired, break genre boundaries, and own what you generate—no strings attached.',
    },
    {
      title: 'Extract stems. Drop into your DAW.',
      description:
        'Export up to 12 time-aligned WAV stems and use them seamlessly in Ableton, Logic, or any DAW. Clean, structured, and ready for pro workflows.',
    },
  ];

  // Pricing plans
  const pricingPlans: PricingPlan[] = [
    {
      name: 'Free Plan',
      price: '$0',
      period: '/month',
      description: 'Our starter plan.',
      features: [
        'Access to v4.5-all',
        '50 credits renew daily (10 songs)',
        'No commercial use',
        'Standard features only',
        'Upload up to 1 min of audio',
        'Shared creation queue',
        'No add-on credit purchases',
      ],
      buttonText: 'Sign Up',
    },
    {
      name: 'Pro Plan',
      price: billingPeriod === 'yearly' ? '$8' : '$10',
      period: '/month',
      description: 'Access to our best models and editing tools.',
      badge: 'Most Popular',
      popular: true,
      features: [
        'Access to latest and most advanced v5 model',
        '2,500 credits (up to 500 songs), refreshes monthly',
        'Commercial use rights for new songs made',
        'Standard + Pro features (personas and advanced editing)',
        'Split songs into up to 12 vocal and instrument stems',
        'Upload up to 8 min of audio',
        'Add new vocals or instrumentals to existing songs',
        'Early access to new features',
        'Ability to purchase add-on credits',
        'Priority queue, up to 10 songs at once',
      ],
      buttonText: 'Subscribe',
    },
    {
      name: 'Premier Plan',
      price: billingPeriod === 'yearly' ? '$24' : '$30',
      period: '/month',
      description: 'Maximum credits and every feature unlocked.',
      badge: 'Best Value',
      features: [
        'Access to Suno Studio',
        'Access to latest and most advanced v5 model',
        '10,000 credits (up to 2,000 songs), refreshes monthly',
        'Commercial use rights for new songs made',
        'Standard + Pro features (personas and advanced editing)',
        'Split songs into up to 12 vocal and instrument stems',
        'Upload up to 8 min of audio',
        'Add new vocals or instrumentals to existing songs',
        'Early access to new features',
        'Ability to purchase add-on credits',
        'Priority queue, up to 10 songs at once',
      ],
      buttonText: 'Subscribe',
    },
  ];

  const handleCreate = () => {
    if (prompt.trim()) {
      // Navigate to music page with prompt
      window.location.href = `${ROUTES.MUSIC}?prompt=${encodeURIComponent(prompt)}`;
    }
  };

  const handleRandomPrompt = () => {
    const randomPrompts = [
      'Make a country song about summer nights',
      'Create an upbeat electronic dance track',
      'Generate a relaxing lo-fi hip hop beat',
      'Write a rock anthem about freedom',
      'Compose a jazz ballad with smooth saxophone',
    ];
    const randomPrompt =
      randomPrompts[Math.floor(Math.random() * randomPrompts.length)];
    setPrompt(randomPrompt);
  };

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden">
      {/* Background Aura Effect */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-orange-950/30 via-black to-black" />
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1200px] h-[1200px] bg-purple-500/20 rounded-full blur-[120px] opacity-50" />
        <div className="absolute top-1/4 right-1/4 w-[600px] h-[600px] bg-pink-500/10 rounded-full blur-[100px] opacity-30" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_rgba(255,255,255,0.01)_0%,_transparent_70%)]" />
      </div>

      {/* Header */}
      <header className="relative z-50 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
            <span className="text-2xl md:text-3xl font-bold text-white tracking-tight uppercase">
              SUNO
            </span>
          </Link>
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="md" className="px-4 py-2 text-white hover:bg-white/10">
              Sign In
            </Button>
            <Button 
              variant="primary" 
              size="md" 
              className="px-4 py-2 bg-gradient-to-r from-orange-400 to-orange-500 hover:from-orange-500 hover:to-orange-600 text-white font-medium shadow-lg shadow-orange-500/20"
            >
              Sign Up
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10">
        {/* Featured Songs Section */}
        <section className="px-6 py-8 -mt-4">
          <div className="max-w-7xl mx-auto">
            <div className="flex gap-4 overflow-x-auto scrollbar-hide pb-4">
              {featuredSongs.map((song) => (
                <div
                  key={song.id}
                  className="flex-shrink-0 w-48 cursor-pointer group"
                >
                  <div className="relative rounded-xl overflow-hidden mb-2 shadow-lg">
                    <div className="w-48 h-48 bg-gradient-to-br from-purple-500/30 to-pink-500/30 flex items-center justify-center group-hover:scale-105 transition-transform duration-300">
                      <Music className="w-16 h-16 text-purple-300/60" />
                    </div>
                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                      <Play className="w-12 h-12 text-white drop-shadow-lg" />
                    </div>
                  </div>
                  <div className="text-sm font-medium truncate">{song.title}</div>
                  <div className="text-xs text-gray-400 truncate">{song.artist}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Hero Section */}
        <section className="px-6 py-8 md:py-12 text-center">
          <div className="max-w-4xl mx-auto">
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-4 md:mb-6 leading-tight">
              {prompt || 'Make any song you can imagine'}
              {prompt && <span className="animate-pulse">|</span>}
            </h1>
            <h2 className="text-lg md:text-xl lg:text-2xl text-gray-300 mb-6 md:mb-8 max-w-2xl mx-auto">
              Start with a simple prompt or dive into our pro editing tools,
              your next track is just a step away.
            </h2>

            {/* Music Creation Input */}
            <div className="max-w-2xl mx-auto mb-6">
              <div className="relative flex items-center gap-2 bg-white/10 backdrop-blur-md rounded-2xl p-3 border border-white/20 hover:border-white/30 transition-colors">
                <Input
                  type="text"
                  placeholder="Chat to make music"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      handleCreate();
                    }
                  }}
                  className="flex-1 bg-transparent border-0 focus:ring-0 text-lg placeholder-gray-400"
                  fullWidth
                />
                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="p-2 hover:bg-white/10"
                    aria-label="Clear"
                    onClick={() => setPrompt('')}
                  >
                    <span className="text-xl font-light">+</span>
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="px-3 py-2 hover:bg-white/10"
                    aria-label="Advanced"
                  >
                    <span className="text-lg font-bold mr-1.5">!</span>
                    <span className="hidden sm:inline text-sm">Advanced</span>
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleRandomPrompt}
                    className="p-2 hover:bg-white/10"
                    aria-label="Generate random prompt"
                  >
                    <Shuffle className="w-5 h-5" />
                  </Button>
                  <Button
                    variant="primary"
                    size="lg"
                    onClick={handleCreate}
                    className="px-6 py-2.5 font-semibold bg-gradient-to-r from-orange-400 to-orange-500 hover:from-orange-500 hover:to-orange-600 text-white shadow-lg shadow-orange-500/20"
                  >
                    <span className="text-lg mr-2">♬</span>
                    Create
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Press Logos - appears twice like in original */}
        <section className="px-6 py-8">
          <div className="max-w-7xl mx-auto">
            <div className="flex flex-wrap items-center justify-center gap-8 opacity-60">
              {pressLogos.map((logo, index) => (
                <div
                  key={index}
                  className="h-8 w-24 bg-white/10 rounded flex items-center justify-center"
                >
                  <span className="text-xs text-gray-400">{logo.name}</span>
                </div>
              ))}
              {/* Duplicate logos as in original */}
              {pressLogos.map((logo, index) => (
                <div
                  key={`dup-${index}`}
                  className="h-8 w-24 bg-white/10 rounded flex items-center justify-center"
                >
                  <span className="text-xs text-gray-400">{logo.name}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Song Quality Section */}
        <section className="px-6 py-16">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="text-4xl md:text-5xl font-bold mb-4">
                Mind blowing song quality
              </h2>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                Whether you have a melody in your head, lyrics you've written,
                or just a feeling you want to hear—Suno makes high-quality music
                creation accessible to all
              </p>
            </div>

            {/* Song Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3 md:gap-4">
              {allSongs.map((song) => (
                <div
                  key={song.id}
                  className="group cursor-pointer"
                  role="button"
                  tabIndex={0}
                >
                  <div className="relative rounded-xl overflow-hidden mb-2 shadow-lg">
                    <div className="w-full aspect-square bg-gradient-to-br from-purple-500/25 to-pink-500/25 flex items-center justify-center group-hover:scale-105 transition-transform duration-300">
                      <Music className="w-10 h-10 md:w-12 md:h-12 text-purple-300/60" />
                    </div>
                    <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-3">
                      <div className="flex items-center gap-1 text-white bg-black/40 px-2 py-1 rounded-full">
                        <Play className="w-3.5 h-3.5 fill-white" />
                        <span className="text-xs font-medium">{song.plays}</span>
                      </div>
                      <button className="flex items-center gap-1 text-white bg-black/40 px-2 py-1 rounded-full hover:bg-black/60 transition-colors">
                        <Heart className="w-3.5 h-3.5 fill-white" />
                        <span className="text-xs font-medium">{song.likes}</span>
                      </button>
                    </div>
                  </div>
                  <div className="text-sm font-medium truncate mb-0.5">
                    {song.title}
                  </div>
                  <div className="text-xs text-gray-400 truncate">
                    {song.artist}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="px-6 py-16 md:py-20 bg-white/[0.03]">
          <div className="max-w-7xl mx-auto">
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold text-center mb-12 md:mb-16">
              Everything you need to make music your way
            </h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
              {/* First row: 3 columns */}
              {features.slice(0, 3).map((feature, index) => (
                <div key={index} className="p-6 md:p-8 rounded-xl bg-white/[0.02] border border-white/5 hover:bg-white/[0.05] transition-colors">
                  <h3 className="text-lg md:text-xl font-semibold mb-3">
                    {feature.title}
                  </h3>
                  <p className="text-gray-300 text-sm md:text-base leading-relaxed">{feature.description}</p>
                </div>
              ))}
              {/* Second row: 3 columns */}
              {features.slice(3, 6).map((feature, index) => (
                <div key={index + 3} className="p-6 md:p-8 rounded-xl bg-white/[0.02] border border-white/5 hover:bg-white/[0.05] transition-colors">
                  <h3 className="text-lg md:text-xl font-semibold mb-3">
                    {feature.title}
                  </h3>
                  <p className="text-gray-300 text-sm md:text-base leading-relaxed">{feature.description}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Pricing Section */}
        <section className="px-6 py-16">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="text-4xl md:text-5xl font-bold mb-4">
                Start making music for free
              </h2>
              <p className="text-xl text-gray-300">
                Select the plan that best fits your needs
              </p>
            </div>

            {/* Billing Toggle */}
            <div className="flex items-center justify-center gap-4 mb-8">
              <div className="flex items-center bg-white/10 rounded-lg p-1">
                <button
                  onClick={() => setBillingPeriod('monthly')}
                  className={`px-4 py-2 rounded-md transition-all ${
                    billingPeriod === 'monthly'
                      ? 'bg-purple-600 text-white shadow-lg'
                      : 'text-gray-300 hover:text-white'
                  }`}
                >
                  Monthly
                </button>
                <button
                  onClick={() => setBillingPeriod('yearly')}
                  className={`px-4 py-2 rounded-md transition-all flex items-center gap-2 ${
                    billingPeriod === 'yearly'
                      ? 'bg-purple-600 text-white shadow-lg'
                      : 'text-gray-300 hover:text-white'
                  }`}
                >
                  Yearly
                  <span className="text-xs bg-green-500 text-white px-2 py-0.5 rounded-full">
                    save 20%
                  </span>
                </button>
              </div>
            </div>

            {/* Pricing Cards */}
            <div className="grid md:grid-cols-3 gap-6">
              {pricingPlans.map((plan, index) => (
                <div
                  key={index}
                  className={`p-6 rounded-2xl border-2 ${
                    plan.popular
                      ? 'border-purple-500 bg-purple-500/10'
                      : 'border-white/20 bg-white/5'
                  }`}
                >
                  {plan.badge && (
                    <div className="mb-4">
                      <span className="inline-block px-3 py-1 bg-purple-500 text-white text-xs font-semibold rounded-full">
                        {plan.badge}
                      </span>
                    </div>
                  )}
                  <h3 className="text-2xl font-bold mb-2">{plan.name}</h3>
                  <p className="text-gray-400 text-sm mb-4">
                    {plan.description}
                  </p>
                  <div className="mb-4">
                    <div className="flex items-baseline gap-1">
                      <span className="text-4xl font-bold">{plan.price}</span>
                      <span className="text-gray-400">{plan.period}</span>
                    </div>
                    {plan.name !== 'Free Plan' && billingPeriod === 'yearly' && (
                      <div className="mt-2 text-sm text-green-400">
                        {plan.name === 'Pro Plan'
                          ? 'Saves $24 by billing yearly!'
                          : 'Saves $72 by billing yearly!'}
                      </div>
                    )}
                    {plan.name !== 'Free Plan' && (
                      <div className="mt-1 text-xs text-gray-500">
                        Taxes calculated at checkout
                      </div>
                    )}
                  </div>
                  <Button
                    variant={plan.popular ? 'primary' : 'outline'}
                    fullWidth
                    className="mb-6"
                  >
                    {plan.buttonText}
                  </Button>
                  <ul className="space-y-3">
                    {plan.features.map((feature, featureIndex) => (
                      <li key={featureIndex} className="flex items-start gap-2">
                        <Check className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                        <span className="text-sm text-gray-300">{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Mobile App Section */}
        <section className="px-6 py-16 bg-white/5">
          <div className="max-w-7xl mx-auto text-center">
            <div className="relative mb-8">
              <div className="absolute inset-0 bg-purple-500/10 rounded-full blur-3xl" />
              <div className="relative">
                <h2 className="text-3xl md:text-4xl font-bold mb-4">
                  The #1 AI music app
                </h2>
                <p className="text-xl text-gray-300 mb-8">
                  Where you can discover, create and share from anywhere because
                  music has no boundaries.
                </p>
                <div className="flex flex-col sm:flex-row items-center justify-center gap-6">
                  <div className="flex items-center gap-4">
                    <div className="text-center">
                      <div className="text-sm text-gray-400 mb-1">
                        Top 10 music app
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="text-2xl font-bold">4.9</span>
                        <span className="text-yellow-400">★★★★★</span>
                      </div>
                      <div className="text-xs text-gray-400">363k+ reviews</div>
                    </div>
                    <Button variant="primary" size="lg">
                      <Download className="w-5 h-5 mr-2" />
                      Download on iPhone
                    </Button>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="text-center">
                      <div className="text-sm text-gray-400 mb-1">
                        Top 10 music app
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="text-2xl font-bold">4.8</span>
                        <span className="text-yellow-400">★★★★★</span>
                      </div>
                      <div className="text-xs text-gray-400">653k+ reviews</div>
                    </div>
                    <Button variant="primary" size="lg">
                      <Download className="w-5 h-5 mr-2" />
                      Download on Android
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Social Media Section */}
        <section className="px-6 py-16">
          <div className="max-w-7xl mx-auto text-center">
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              Explore and get inspired
            </h2>
            <p className="text-xl text-gray-300 mb-8">
              Join the #1 AI music generator. Create songs, remix tracks, make
              beats, and share your music with millions — free forever.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-6 mb-8">
              {[
                '@timbaland',
                '@spellspellspell',
                '@nickfloats',
                '@milesmusickid',
                '@devanibiza',
                '@techguyver',
              ].map((handle) => (
                <div
                  key={handle}
                  className="w-24 h-24 rounded-full bg-gradient-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center cursor-pointer hover:scale-110 transition-transform"
                >
                  <Music className="w-12 h-12 text-purple-400/50" />
                </div>
              ))}
            </div>
            <Button variant="primary" size="lg">
              Sign up for free
            </Button>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="relative z-10 px-6 py-12 border-t border-white/10">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-4 gap-8 mb-8">
            <div>
              <h4 className="font-semibold mb-4">Brand</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>
                  <Link href="/about" className="hover:text-white">
                    About
                  </Link>
                </li>
                <li>
                  <Link href="/careers" className="hover:text-white">
                    Work at Suno
                  </Link>
                </li>
                <li>
                  <Link href="/blog" className="hover:text-white">
                    Blog
                  </Link>
                </li>
                <li>
                  <Link href="/pricing" className="hover:text-white">
                    Pricing
                  </Link>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Support</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>
                  <Link href="/help" className="hover:text-white">
                    Help
                  </Link>
                </li>
                <li>
                  <Link href="/contact" className="hover:text-white">
                    Contact Us
                  </Link>
                </li>
                <li>
                  <Link href="/faq" className="hover:text-white">
                    FAQs
                  </Link>
                </li>
                <li>
                  <Link href="/terms" className="hover:text-white">
                    Terms of Service
                  </Link>
                </li>
                <li>
                  <Link href="/privacy" className="hover:text-white">
                    Privacy Policy
                  </Link>
                </li>
              </ul>
            </div>
          </div>
          <div className="flex items-center justify-between pt-8 border-t border-white/10">
            <p className="text-sm text-gray-400">© 2025 Suno, Inc.</p>
            <div className="flex items-center gap-4">
              <Link
                href="https://twitter.com/sunomusic"
                className="text-gray-400 hover:text-white"
                aria-label="Twitter"
              >
                <Twitter className="w-5 h-5" />
              </Link>
              <Link
                href="https://instagram.com/sunomusic"
                className="text-gray-400 hover:text-white"
                aria-label="Instagram"
              >
                <Instagram className="w-5 h-5" />
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
