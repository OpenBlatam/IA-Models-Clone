import { ParallaxScroll } from "@/components/parallax-scroll"
import { motion } from "framer-motion"
import { useRouter } from "next/navigation"

export function MarketingParallax() {
  const router = useRouter();
  return (
    <div className="relative min-h-screen bg-black text-white overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-black via-purple-900/20 to-black" />
      
      {/* Main content */}
      <div className="relative z-10 container mx-auto px-4 py-20">
        <div className="space-y-20">
          {/* First section */}
          <ParallaxScroll speed={0.5}>
            <div className="text-center space-y-4">
              <motion.h2 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
                className="text-5xl md:text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-600"
              >
                Transform Your Marketing
              </motion.h2>
              <motion.p 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                className="text-xl text-gray-300 max-w-2xl mx-auto"
              >
                Experience the future of marketing with our AI-powered platform
              </motion.p>
            </div>
          </ParallaxScroll>

          {/* Second section */}
          <ParallaxScroll speed={0.3}>
            <div className="grid md:grid-cols-2 gap-12 items-center">
              <motion.div 
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8 }}
                className="space-y-4"
              >
                <h3 className="text-3xl font-bold">Smart Analytics</h3>
                <p className="text-gray-300">
                  Get real-time insights and analytics to optimize your marketing campaigns
                </p>
              </motion.div>
              <motion.div 
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8 }}
                className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 p-8 rounded-2xl"
              >
                <div className="aspect-video bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg" />
              </motion.div>
            </div>
          </ParallaxScroll>

          {/* Third section */}
          <ParallaxScroll speed={0.4}>
            <div className="grid md:grid-cols-2 gap-12 items-center">
              <motion.div 
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8 }}
                className="order-2 md:order-1 bg-gradient-to-r from-purple-500/20 to-pink-500/20 p-8 rounded-2xl"
              >
                <div className="aspect-video bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg" />
              </motion.div>
              <motion.div 
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8 }}
                className="space-y-4 order-1 md:order-2"
              >
                <h3 className="text-3xl font-bold">AI-Powered Insights</h3>
                <p className="text-gray-300">
                  Leverage artificial intelligence to predict trends and optimize your strategy
                </p>
              </motion.div>
            </div>
          </ParallaxScroll>

          {/* Fourth section */}
          <ParallaxScroll speed={0.6}>
            <div className="text-center space-y-4">
              <motion.h2 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
                className="text-4xl md:text-6xl font-bold"
              >
                Ready to Get Started?
              </motion.h2>
              <motion.p 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                className="text-xl text-gray-300 max-w-2xl mx-auto"
              >
                Join thousands of marketers who are already transforming their business
              </motion.p>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.4 }}
              >
                <button 
                  onClick={() => router.push("/signup")}
                  className="inline-block mt-8 px-8 py-4 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full text-white font-semibold hover:opacity-90 transition-opacity"
                >
                  Sign up
                </button>
              </motion.div>
            </div>
          </ParallaxScroll>
        </div>
      </div>
    </div>
  )
}   