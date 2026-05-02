const { PrismaClient } = require("@prisma/client");

const prisma = new PrismaClient();

async function main() {
  // Create Marketing con IA lesson
  const marketingLesson = await prisma.lesson.create({
    data: {
      title: "Marketing con IA",
      description: "Aprende a utilizar la inteligencia artificial para mejorar tus estrategias de marketing.",
      difficulty: "Intermedio",
      category: "Marketing",
      requiredLevel: 1,
      experienceReward: 20,
      exercises: {
        create: [
          {
            type: "multiple_choice",
            question: "¿Cuál es el principal beneficio de usar IA en marketing?",
            options: [
              "Automatizar todas las decisiones",
              "Personalización y segmentación avanzada",
              "Reducir costos de personal",
              "Eliminar la creatividad humana"
            ],
            correctAnswer: "Personalización y segmentación avanzada",
            points: 25
          },
          {
            type: "multiple_choice",
            question: "¿Qué herramienta de IA es más útil para análisis de sentimiento?",
            options: [
              "ChatGPT",
              "DALL-E",
              "BERT",
              "Stable Diffusion"
            ],
            correctAnswer: "BERT",
            points: 25
          },
          {
            type: "multiple_choice",
            question: "¿Cómo puede la IA mejorar la segmentación de audiencia?",
            options: [
              "Solo por edad y género",
              "Solo por ubicación",
              "Analizando patrones de comportamiento complejos",
              "Solo por ingresos"
            ],
            correctAnswer: "Analizando patrones de comportamiento complejos",
            points: 25
          },
          {
            type: "multiple_choice",
            question: "¿Cuál es el mejor uso de la IA en marketing de contenidos?",
            options: [
              "Generar todo el contenido automáticamente",
              "Optimizar y personalizar contenido existente",
              "Reemplazar escritores humanos",
              "Solo para correos electrónicos"
            ],
            correctAnswer: "Optimizar y personalizar contenido existente",
            points: 25
          }
        ]
      }
    },
    include: {
      exercises: true
    }
  });



  // Create 10 basic marketing lessons for the "Principiante" path
  const basicMarketingLessons = [
    {
      title: "Introducción al Marketing",
      description: "Aprende los conceptos fundamentales del marketing y su importancia en el negocio.",
      difficulty: "Principiante",
      category: "Marketing",
      requiredLevel: 1,
      experienceReward: 100,
      exercises: {
        create: [
          {
            type: "multiple_choice",
            question: "¿Qué es el marketing?",
            options: [
              "Solo publicidad",
              "Un conjunto de actividades para crear, comunicar y entregar valor a los clientes",
              "Solo ventas",
              "Solo redes sociales"
            ],
            correctAnswer: "Un conjunto de actividades para crear, comunicar y entregar valor a los clientes",
            points: 25
          }
        ]
      }
    },
    {
      title: "Segmentación de Mercado",
      description: "Aprende a identificar y segmentar tu mercado objetivo.",
      difficulty: "Principiante",
      category: "Marketing",
      requiredLevel: 1,
      experienceReward: 100,
      exercises: {
        create: [
          {
            type: "multiple_choice",
            question: "¿Qué es la segmentación de mercado?",
            options: [
              "Dividir el mercado en grupos homogéneos",
              "Vender a todos los clientes",
              "Ignorar las diferencias entre clientes",
              "Solo enfocarse en un tipo de cliente"
            ],
            correctAnswer: "Dividir el mercado en grupos homogéneos",
            points: 25
          }
        ]
      }
    },
    {
      title: "Posicionamiento de Marca",
      description: "Aprende a posicionar tu marca en la mente de los consumidores.",
      difficulty: "Principiante",
      category: "Marketing",
      requiredLevel: 1,
      experienceReward: 100,
      exercises: {
        create: [
          {
            type: "multiple_choice",
            question: "¿Qué es el posicionamiento de marca?",
            options: [
              "Solo el logo de la empresa",
              "La percepción que los consumidores tienen de tu marca",
              "Solo el nombre de la empresa",
              "Solo el eslogan"
            ],
            correctAnswer: "La percepción que los consumidores tienen de tu marca",
            points: 25
          }
        ]
      }
    },
    {
      title: "Marketing Mix (4P's)",
      description: "Aprende sobre Producto, Precio, Plaza y Promoción.",
      difficulty: "Principiante",
      category: "Marketing",
      requiredLevel: 1,
      experienceReward: 100,
      exercises: {
        create: [
          {
            type: "multiple_choice",
            question: "¿Cuáles son las 4P's del marketing?",
            options: [
              "Producto, Precio, Plaza, Promoción",
              "Personas, Proceso, Producto, Precio",
              "Publicidad, Precio, Producto, Plaza",
              "Promoción, Plaza, Personas, Producto"
            ],
            correctAnswer: "Producto, Precio, Plaza, Promoción",
            points: 25
          }
        ]
      }
    },
    {
      title: "Investigación de Mercado",
      description: "Aprende a realizar investigación de mercado efectiva.",
      difficulty: "Principiante",
      category: "Marketing",
      requiredLevel: 1,
      experienceReward: 100,
      exercises: {
        create: [
          {
            type: "multiple_choice",
            question: "¿Qué es la investigación de mercado?",
            options: [
              "Solo preguntar a amigos",
              "Recopilar y analizar información sobre el mercado",
              "Ignorar la competencia",
              "Solo mirar redes sociales"
            ],
            correctAnswer: "Recopilar y analizar información sobre el mercado",
            points: 25
          }
        ]
      }
    },
    {
      title: "Marketing Digital",
      description: "Aprende los conceptos básicos del marketing digital.",
      difficulty: "Principiante",
      category: "Marketing",
      requiredLevel: 1,
      experienceReward: 100,
      exercises: {
        create: [
          {
            type: "multiple_choice",
            question: "¿Qué es el marketing digital?",
            options: [
              "Solo redes sociales",
              "Estrategias de marketing en canales digitales",
              "Solo email marketing",
              "Solo publicidad en Google"
            ],
            correctAnswer: "Estrategias de marketing en canales digitales",
            points: 25
          }
        ]
      }
    },
    {
      title: "Content Marketing",
      description: "Aprende a crear y distribuir contenido valioso.",
      difficulty: "Principiante",
      category: "Marketing",
      requiredLevel: 1,
      experienceReward: 100,
      exercises: {
        create: [
          {
            type: "multiple_choice",
            question: "¿Qué es el content marketing?",
            options: [
              "Solo escribir blogs",
              "Crear y distribuir contenido valioso para atraer audiencia",
              "Solo videos",
              "Solo infografías"
            ],
            correctAnswer: "Crear y distribuir contenido valioso para atraer audiencia",
            points: 25
          }
        ]
      }
    },
    {
      title: "Email Marketing",
      description: "Aprende a utilizar el email marketing efectivamente.",
      difficulty: "Principiante",
      category: "Marketing",
      requiredLevel: 1,
      experienceReward: 100,
      exercises: {
        create: [
          {
            type: "multiple_choice",
            question: "¿Qué es el email marketing?",
            options: [
              "Solo enviar emails",
              "Estrategias de marketing a través del correo electrónico",
              "Solo newsletters",
              "Solo promociones"
            ],
            correctAnswer: "Estrategias de marketing a través del correo electrónico",
            points: 25
          }
        ]
      }
    },
    {
      title: "Social Media Marketing",
      description: "Aprende a utilizar las redes sociales para marketing.",
      difficulty: "Principiante",
      category: "Marketing",
      requiredLevel: 1,
      experienceReward: 100,
      exercises: {
        create: [
          {
            type: "multiple_choice",
            question: "¿Qué es el social media marketing?",
            options: [
              "Solo publicar en redes",
              "Estrategias de marketing en redes sociales",
              "Solo Facebook",
              "Solo Instagram"
            ],
            correctAnswer: "Estrategias de marketing en redes sociales",
            points: 25
          }
        ]
      }
    },
    {
      title: "Analítica Web",
      description: "Aprende a medir y analizar el rendimiento de tu marketing digital.",
      difficulty: "Principiante",
      category: "Marketing",
      requiredLevel: 1,
      experienceReward: 100,
      exercises: {
        create: [
          {
            type: "multiple_choice",
            question: "¿Qué es la analítica web?",
            options: [
              "Solo contar visitas",
              "Medir y analizar el comportamiento de los usuarios en tu sitio web",
              "Solo Google Analytics",
              "Solo métricas de redes sociales"
            ],
            correctAnswer: "Medir y analizar el comportamiento de los usuarios en tu sitio web",
            points: 25
          }
        ]
      }
    }
  ];

  for (const lesson of basicMarketingLessons) {
    await prisma.lesson.create({
      data: lesson,
      include: {
        exercises: true
      }
    });
  }


}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });  