"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Trophy, Brain, Target, Users, BarChart2, Mail, MessageSquare, ShoppingBag, Sparkles, Bird } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import * as Dialog from "@radix-ui/react-dialog"
import { toast } from "sonner"
import { Howl } from "howler"
import Confetti from "react-confetti"
import { useGesture } from "@use-gesture/react"
import { useSpring, animated } from "@react-spring/web"
import gsap from "gsap"
import { Particles } from "react-particles"
import { loadSlim } from "tsparticles-slim"
import Tilt from "react-parallax-tilt"
import CountUp from "react-countup"

interface Obstacle {
  x: number
  gapY: number
  passed: boolean
  type: 'social' | 'email' | 'content' | 'analytics'
}

interface GameState {
  isPlaying: boolean
  score: number
  highScore: number
  gameOver: boolean
}

// Particle effect for visual feedback
interface Particle {
  x: number
  y: number
  vx: number
  vy: number
  life: number
  color: string
}

const MARKETING_TYPES = {
  social: {
    color: '#3b82f6',
    icon: <MessageSquare className="h-4 w-4" />,
    text: 'Social Media'
  },
  email: {
    color: '#ef4444',
    icon: <Mail className="h-4 w-4" />,
    text: 'Email Marketing'
  },
  content: {
    color: '#22c55e',
    icon: <ShoppingBag className="h-4 w-4" />,
    text: 'Content Marketing'
  },
  analytics: {
    color: '#8b5cf6',
    icon: <BarChart2 className="h-4 w-4" />,
    text: 'Analytics'
  }
}

// Sound effects with volume control
const sounds = {
  flap: new Howl({ 
    src: ['/sounds/flap.mp3'],
    volume: 0.3
  }),
  point: new Howl({ 
    src: ['/sounds/point.mp3'],
    volume: 0.4
  }),
  hit: new Howl({ 
    src: ['/sounds/hit.mp3'],
    volume: 0.5
  }),
  gameOver: new Howl({ 
    src: ['/sounds/game-over.mp3'],
    volume: 0.6
  }),
  background: new Howl({ 
    src: ['/sounds/background.mp3'],
    loop: true,
    volume: 0.2
  })
}

export function MarketingFlappy() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [gameState, setGameState] = useState<GameState>({
    isPlaying: false,
    score: 0,
    highScore: 0,
    gameOver: false
  })
  const [showConfetti, setShowConfetti] = useState(false)
  const [windowSize, setWindowSize] = useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 0,
    height: typeof window !== 'undefined' ? window.innerHeight : 0
  })
  const [particles, setParticles] = useState<Particle[]>([])
  const [isMuted, setIsMuted] = useState(false)
  const gameContainerRef = useRef<HTMLDivElement>(null)
  const birdRef = useRef<HTMLDivElement>(null)
  const [spring, api] = useSpring(() => ({
    from: { y: 250, rotate: 0 },
    config: { tension: 300, friction: 10 }
  }))
  const [particlesInit, setParticlesInit] = useState(false)
  const [showScore, setShowScore] = useState(false)

  // Game constants
  const GRAVITY = 0.5
  const FLAP_FORCE = -8
  const PIPE_SPEED = 2
  const PIPE_SPAWN_INTERVAL = 1500
  const PIPE_GAP = 150
  const BIRD_SIZE = 30

  // Game state refs
  const birdY = useRef(250)
  const birdVelocity = useRef(0)
  const obstacles = useRef<Obstacle[]>([])
  const lastPipeSpawn = useRef(0)
  const animationFrameId = useRef<number>()
  const backgroundOffset = useRef(0)

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight
      })
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  // GSAP animations
  const animateBird = useCallback((y: number) => {
    if (birdRef.current) {
      gsap.to(birdRef.current, {
        y: y,
        rotation: y > birdY.current ? 30 : -30,
        duration: 0.3,
        ease: "power2.out"
      })
    }
  }, [])

  // Enhanced gesture controls
  const bind = useGesture({
    onDrag: ({ movement: [x, y], first, last }) => {
      if (first) {
        handleFlap()
      }
    },
    onWheel: ({ direction: [x, y] }) => {
      if (y < 0) {
        handleFlap()
      }
    },
    onMove: ({ xy: [x, y] }) => {
      if (gameState.isPlaying && gameContainerRef.current) {
        const rect = gameContainerRef.current.getBoundingClientRect()
        const relativeY = y - rect.top
        if (relativeY < birdY.current - 50) {
          handleFlap()
        }
      }
    }
  })

  // Create particles for visual feedback
  const createParticles = useCallback((x: number, y: number, color: string) => {
    const newParticles: Particle[] = []
    for (let i = 0; i < 10; i++) {
      newParticles.push({
        x,
        y,
        vx: (Math.random() - 0.5) * 5,
        vy: (Math.random() - 0.5) * 5,
        life: 1,
        color
      })
    }
    setParticles(prev => [...prev, ...newParticles])
  }, [])

  // Update particles
  useEffect(() => {
    if (particles.length === 0) return

    const interval = setInterval(() => {
      setParticles(prev => 
        prev
          .map(p => ({
            ...p,
            x: p.x + p.vx,
            y: p.y + p.vy,
            life: p.life - 0.02
          }))
          .filter(p => p.life > 0)
      )
    }, 16)

    return () => clearInterval(interval)
  }, [particles])

  // Toggle sound
  const toggleSound = useCallback(() => {
    setIsMuted(prev => {
      const newMuted = !prev
      Object.values(sounds).forEach(sound => {
        sound.mute(newMuted)
      })
      return newMuted
    })
  }, [])

  const handleScore = useCallback((newScore: number) => {
    setGameState(prev => ({
      ...prev,
      score: newScore,
      highScore: Math.max(newScore, prev.highScore)
    }))

    if (!isMuted) sounds.point.play()
    createParticles(100, birdY.current, '#22c55e')

    // Show confetti for milestones
    if (newScore === 5 || newScore === 10 || newScore % 10 === 0) {
      setShowConfetti(true)
      setTimeout(() => setShowConfetti(false), 5000)
    }

    // Show toast notifications for milestones
    if (newScore === 5) {
      toast.success("¡5 puntos! ¡Estás en camino de ser un experto en marketing!")
    } else if (newScore === 10) {
      toast.success("¡10 puntos! ¡Eres un maestro del marketing!")
    } else if (newScore % 5 === 0 && newScore > 0) {
      toast.success(`¡${newScore} puntos! ¡Sigue así!`)
    }
  }, [isMuted, createParticles])

  const startGame = useCallback(() => {
    setGameState({
      isPlaying: true,
      score: 0,
      highScore: gameState.highScore,
      gameOver: false
    })
    birdY.current = 250
    birdVelocity.current = 0
    obstacles.current = []
    lastPipeSpawn.current = 0
    backgroundOffset.current = 0
    sounds.background.play()
    toast.info("¡Comienza el juego! Usa la barra espaciadora o haz clic para volar")
  }, [gameState.highScore])

  const handleFlap = useCallback(() => {
    if (!gameState.isPlaying) {
      startGame()
    }
    birdVelocity.current = FLAP_FORCE
    if (!isMuted) sounds.flap.play()
    createParticles(100, birdY.current, '#3b82f6')
    
    // Animate bird with spring
    api.start({
      from: { y: birdY.current, rotate: 30 },
      to: { y: birdY.current - 50, rotate: -30 },
      reset: true
    })
  }, [gameState.isPlaying, startGame, isMuted, createParticles, api])

  const handleGameOver = useCallback(() => {
    if (!isMuted) {
      Object.values(sounds).forEach(sound => {
        sound.stop()
      })
      sounds.hit.play()
      setTimeout(() => {
        sounds.gameOver.play()
      }, 500)
    }
    
    // Animate bird falling
    if (birdRef.current) {
      gsap.to(birdRef.current, {
        y: 600,
        rotation: 90,
        duration: 1,
        ease: "power2.in"
      })
    }
    
    createParticles(100, birdY.current, '#ef4444')
    setGameState(prev => ({
      ...prev,
      isPlaying: false,
      gameOver: true
    }))
  }, [isMuted, createParticles])

  const spawnPipe = useCallback(() => {
    const types = Object.keys(MARKETING_TYPES) as Array<keyof typeof MARKETING_TYPES>
    const randomType = types[Math.floor(Math.random() * types.length)]
    const gapY = Math.random() * (400 - PIPE_GAP)
    obstacles.current.push({
      x: 800,
      gapY,
      passed: false,
      type: randomType
    })
  }, [])

  const drawBackground = useCallback((ctx: CanvasRenderingContext2D, width: number, height: number) => {
    // Draw gradient background
    const gradient = ctx.createLinearGradient(0, 0, 0, height)
    gradient.addColorStop(0, '#f0f9ff')
    gradient.addColorStop(1, '#e0f2fe')
    ctx.fillStyle = gradient
    ctx.fillRect(0, 0, width, height)

    // Draw moving clouds
    backgroundOffset.current = (backgroundOffset.current + 0.5) % width
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
    for (let i = 0; i < 5; i++) {
      const x = (backgroundOffset.current + i * 200) % width
      ctx.beginPath()
      ctx.arc(x, 50 + i * 30, 20, 0, Math.PI * 2)
      ctx.arc(x + 15, 50 + i * 30, 15, 0, Math.PI * 2)
      ctx.arc(x + 30, 50 + i * 30, 20, 0, Math.PI * 2)
      ctx.fill()
    }
  }, [])

  const drawBird = useCallback((ctx: CanvasRenderingContext2D, y: number) => {
    // Draw bird body
    ctx.fillStyle = '#3b82f6'
    ctx.beginPath()
    ctx.arc(100, y, BIRD_SIZE / 2, 0, Math.PI * 2)
    ctx.fill()

    // Draw bird eye
    ctx.fillStyle = 'white'
    ctx.beginPath()
    ctx.arc(110, y - 5, 5, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = 'black'
    ctx.beginPath()
    ctx.arc(110, y - 5, 2, 0, Math.PI * 2)
    ctx.fill()

    // Draw bird wing
    ctx.fillStyle = '#2563eb'
    ctx.beginPath()
    ctx.ellipse(90, y + 5, 10, 5, Math.PI / 4, 0, Math.PI * 2)
    ctx.fill()
  }, [])

  const drawObstacle = useCallback((ctx: CanvasRenderingContext2D, obstacle: Obstacle) => {
    const type = MARKETING_TYPES[obstacle.type]
    
    // Draw top pipe
    ctx.fillStyle = type.color
    ctx.fillRect(obstacle.x, 0, 80, obstacle.gapY)
    
    // Draw pipe cap
    ctx.fillStyle = type.color
    ctx.fillRect(obstacle.x - 5, obstacle.gapY - 20, 90, 20)
    
    // Draw bottom pipe
    ctx.fillStyle = type.color
    ctx.fillRect(obstacle.x, obstacle.gapY + PIPE_GAP, 80, 800)
    
    // Draw pipe cap
    ctx.fillStyle = type.color
    ctx.fillRect(obstacle.x - 5, obstacle.gapY + PIPE_GAP, 90, 20)

    // Draw marketing type text
    ctx.fillStyle = 'white'
    ctx.font = '12px Arial'
    ctx.textAlign = 'center'
    ctx.fillText(type.text, obstacle.x + 40, obstacle.gapY - 10)
    ctx.fillText(type.text, obstacle.x + 40, obstacle.gapY + PIPE_GAP + 15)
  }, [])

  const updateGame = useCallback(() => {
    if (!gameState.isPlaying) return

    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Update bird
    birdVelocity.current += GRAVITY
    birdY.current += birdVelocity.current

    // Update pipes
    const now = Date.now()
    if (now - lastPipeSpawn.current > PIPE_SPAWN_INTERVAL) {
      spawnPipe()
      lastPipeSpawn.current = now
    }

    obstacles.current = obstacles.current.filter(obstacle => {
      obstacle.x -= PIPE_SPEED

      // Check if bird passed the pipe
      if (!obstacle.passed && obstacle.x < 100) {
        obstacle.passed = true
        handleScore(gameState.score + 1)
      }

      return obstacle.x > -100
    })

    // Check collisions
    const birdRect = {
      x: 100,
      y: birdY.current,
      width: BIRD_SIZE,
      height: BIRD_SIZE
    }

    const collision = obstacles.current.some(obstacle => {
      const topPipe = {
        x: obstacle.x,
        y: 0,
        width: 80,
        height: obstacle.gapY
      }
      const bottomPipe = {
        x: obstacle.x,
        y: obstacle.gapY + PIPE_GAP,
        width: 80,
        height: 800
      }

      return (
        birdRect.x < topPipe.x + topPipe.width &&
        birdRect.x + birdRect.width > topPipe.x &&
        birdRect.y < topPipe.y + topPipe.height &&
        birdRect.y + birdRect.height > topPipe.y
      ) || (
        birdRect.x < bottomPipe.x + bottomPipe.width &&
        birdRect.x + birdRect.width > bottomPipe.x &&
        birdRect.y < bottomPipe.y + bottomPipe.height &&
        birdRect.y + birdRect.height > bottomPipe.y
      )
    })

    if (collision || birdY.current < 0 || birdY.current > 600) {
      handleGameOver()
      return
    }

    // Draw everything
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw background
    drawBackground(ctx, canvas.width, canvas.height)

    // Draw obstacles
    obstacles.current.forEach(obstacle => drawObstacle(ctx, obstacle))

    // Draw bird
    drawBird(ctx, birdY.current)

    // Draw score
    ctx.fillStyle = '#1e293b'
    ctx.font = '24px Arial'
    ctx.textAlign = 'left'
    ctx.fillText(`Score: ${gameState.score}`, 20, 40)
    ctx.fillText(`High Score: ${gameState.highScore}`, 20, 70)

    animationFrameId.current = requestAnimationFrame(updateGame)
  }, [gameState.isPlaying, spawnPipe, drawBackground, drawBird, drawObstacle, handleScore, handleGameOver])

  useEffect(() => {
    if (gameState.isPlaying) {
      animationFrameId.current = requestAnimationFrame(updateGame)
    }
    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current)
      }
    }
  }, [gameState.isPlaying, updateGame])

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        handleFlap()
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [handleFlap])

  // Cleanup sounds when component unmounts
  useEffect(() => {
    return () => {
      Object.values(sounds).forEach(sound => {
        sound.stop()
      })
    }
  }, [])

  // Initialize particles
  const initParticles = useCallback(async (engine: any) => {
    await loadSlim(engine)
    setParticlesInit(true)
  }, [])

  // Particles configuration
  const particlesConfig = {
    particles: {
      number: {
        value: 50,
        density: {
          enable: true,
          value_area: 800
        }
      },
      color: {
        value: "#3b82f6"
      },
      shape: {
        type: "circle"
      },
      opacity: {
        value: 0.5,
        random: true
      },
      size: {
        value: 3,
        random: true
      },
      move: {
        enable: true,
        speed: 1,
        direction: "none" as const,
        random: true,
        outModes: {
          default: "out"
        }
      }
    },
    interactivity: {
      events: {
        onHover: {
          enable: true,
          mode: "repulse"
        }
      }
    }
  } as const

  if (gameState.gameOver) {
    return (
      <>
        {showConfetti && (
          <Confetti
            width={windowSize.width}
            height={windowSize.height}
            recycle={false}
            numberOfPieces={200}
          />
        )}
        <Dialog.Root open={gameState.gameOver}>
          <Dialog.Portal>
            <Dialog.Overlay className="fixed inset-0 bg-black/50 backdrop-blur-sm" />
            <Dialog.Content className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
              >
                <Card className="p-6 w-[400px]">
                  <div className="text-center space-y-4">
                    <motion.div
                      initial={{ y: -20, rotate: -10 }}
                      animate={{ y: 0, rotate: 0 }}
                      transition={{ type: "spring", stiffness: 200 }}
                    >
                      <Trophy className="h-16 w-16 mx-auto text-yellow-500" />
                    </motion.div>
                    <h2 className="text-2xl font-bold">¡Juego Terminado!</h2>
                    <motion.p
                      initial={{ scale: 0.8 }}
                      animate={{ scale: 1 }}
                      className="text-xl"
                    >
                      Puntuación: {gameState.score}
                    </motion.p>
                    <p className="text-muted-foreground">
                      {gameState.score > 10
                        ? "¡Excelente! ¡Eres un experto en marketing!"
                        : gameState.score > 5
                          ? "¡Buen trabajo! Sigue practicando."
                          : "¡Sigue intentándolo! La práctica hace al maestro."}
                    </p>
                    <motion.div
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <Button onClick={startGame} className="mt-4 w-full">
                        Jugar de nuevo
                      </Button>
                    </motion.div>
                  </div>
                </Card>
              </motion.div>
            </Dialog.Content>
          </Dialog.Portal>
        </Dialog.Root>
      </>
    )
  }

  return (
    <Card className="p-6">
      <div className="space-y-6">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between"
        >
          <h3 className="text-lg font-semibold">
            Marketing Flappy
          </h3>
          <div className="flex items-center gap-4">
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={toggleSound}
              className="p-2 rounded-full hover:bg-muted"
            >
              {isMuted ? (
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="1" y1="1" x2="23" y2="23"></line>
                  <path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"></path>
                  <path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23"></path>
                  <line x1="12" y1="19" x2="12" y2="23"></line>
                  <line x1="8" y1="23" x2="16" y2="23"></line>
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                  <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
                  <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
                </svg>
              )}
            </motion.button>
            <Tilt
              tiltMaxAngleX={10}
              tiltMaxAngleY={10}
              perspective={1000}
              glareEnable={true}
              glareMaxOpacity={0.3}
              glareColor="#ffffff"
              glarePosition="all"
              glareBorderRadius="20px"
            >
              <motion.div
                initial={{ scale: 0.8 }}
                animate={{ scale: 1 }}
                className="text-sm text-muted-foreground bg-muted/50 p-2 rounded-lg"
              >
                Puntuación más alta:{" "}
                <CountUp
                  end={gameState.highScore}
                  duration={2}
                  separator=","
                  className="font-bold text-primary"
                />
              </motion.div>
            </Tilt>
          </div>
        </motion.div>

        <div className="relative" ref={gameContainerRef} {...bind()}>
          <canvas
            ref={canvasRef}
            width={800}
            height={600}
            className="w-full h-[400px] bg-sky-50 rounded-lg cursor-pointer"
            onClick={handleFlap}
          />
          
          {/* Particles background */}
          {particlesInit && (
            <Particles
              id="tsparticles"
              init={initParticles}
              options={particlesConfig}
              className="absolute inset-0 pointer-events-none"
            />
          )}
          
          {/* Animated bird */}
          <animated.div
            ref={birdRef}
            style={{
              position: 'absolute',
              left: 100,
              top: 0,
              transform: spring.y.to(y => `translateY(${y}px) rotate(${spring.rotate}deg)`),
              zIndex: 10
            }}
          >
            <Tilt
              tiltMaxAngleX={15}
              tiltMaxAngleY={15}
              perspective={1000}
              glareEnable={true}
              glareMaxOpacity={0.5}
              glareColor="#ffffff"
              glarePosition="all"
              glareBorderRadius="50%"
            >
              <Bird className="h-8 w-8 text-blue-500" />
            </Tilt>
          </animated.div>

          {/* Particles */}
          <svg
            className="absolute inset-0 pointer-events-none"
            width="100%"
            height="100%"
          >
            {particles.map((p, i) => (
              <animated.circle
                key={i}
                cx={p.x}
                cy={p.y}
                r={3}
                fill={p.color}
                opacity={p.life}
              />
            ))}
          </svg>

          <AnimatePresence>
            {!gameState.isPlaying && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-lg"
              >
                <Tilt
                  tiltMaxAngleX={10}
                  tiltMaxAngleY={10}
                  perspective={1000}
                  glareEnable={true}
                  glareMaxOpacity={0.3}
                  glareColor="#ffffff"
                  glarePosition="all"
                  glareBorderRadius="20px"
                >
                  <motion.div
                    initial={{ scale: 0.8, y: 20 }}
                    animate={{ scale: 1, y: 0 }}
                    className="text-center text-white bg-black/30 p-8 rounded-xl backdrop-blur-sm"
                  >
                    <motion.div
                      animate={{
                        y: [0, -10, 0],
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        ease: "easeInOut",
                      }}
                    >
                      <Bird className="h-12 w-12 mx-auto mb-4" />
                    </motion.div>
                    <motion.p
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.2 }}
                      className="text-xl font-bold mb-2"
                    >
                      ¡Haz clic para comenzar!
                    </motion.p>
                    <motion.p
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.4 }}
                      className="text-sm"
                    >
                      Usa la barra espaciadora, haz clic o desliza para volar
                    </motion.p>
                  </motion.div>
                </Tilt>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="text-center text-sm text-muted-foreground"
        >
          <p>Evita los obstáculos y recoge puntos</p>
          <p>¡Cada obstáculo superado es un concepto de marketing aprendido!</p>
        </motion.div>

        {gameState.isPlaying && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mt-4"
          >
            <Progress value={(gameState.score / 20) * 100} className="h-2" />
            <p className="text-sm text-muted-foreground mt-2 text-center">
              Progreso:{" "}
              <CountUp
                end={gameState.score}
                duration={0.5}
                className="font-bold text-primary"
              />{" "}
              / 20
            </p>
          </motion.div>
        )}
      </div>
    </Card>
  )
}                                                                                                                         