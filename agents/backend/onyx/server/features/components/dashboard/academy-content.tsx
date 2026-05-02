'use client';

import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Icons } from "@/components/shared/icons";
import { cn } from "@/lib/utils";
import { useState } from "react";

interface LearningPath {
  id: string;
  title: string;
  description: string;
  icon: keyof typeof Icons;
  color: string;
  lessons: number;
  xp: number;
}

interface AcademyContentProps {
  learningPaths: LearningPath[];
}

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 }
};

export function AcademyContent({ learningPaths }: AcademyContentProps) {
  const [selectedPath, setSelectedPath] = useState<LearningPath | null>(null);
  const [score, setScore] = useState(0);
  const [globalScore, setGlobalScore] = useState(0);
  const [feedback, setFeedback] = useState<string | null>(null);

  const handlePathClick = (path: LearningPath) => {
    setSelectedPath(path);
  };

  const handleAnswer = (answer: string, correctAnswer: string) => {
    if (answer === correctAnswer) {
      setScore(score + 25);
      setGlobalScore(globalScore + 25);
      setFeedback("¡Correcto! Has ganado 25 XP.");
    } else {
      setFeedback(`Incorrecto. La respuesta correcta es: ${correctAnswer}`);
    }
  };

  if (selectedPath) {
    return (
      <div className="p-8 pt-6 bg-white rounded-lg shadow-lg transform transition-transform hover:scale-105">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">{selectedPath.title}</h2>
        <p className="text-muted-foreground mb-4 text-gray-600">{selectedPath.description}</p>
        <div className="flex items-center justify-between text-xs mb-4">
          <div className="flex items-center gap-2">
            <Icons.bookOpen className="h-3 w-3 text-muted-foreground" />
            <span>{selectedPath.lessons} lecciones</span>
          </div>
          <div className="flex items-center gap-2">
            <Icons.star className="h-3 w-3 text-yellow-500" />
            <span>{selectedPath.xp} XP</span>
          </div>
        </div>
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800">Introducción al Marketing</h3>
          <p className="text-gray-600">Bienvenido al mundo del marketing, donde la creatividad y la estrategia se encuentran. En esta sección, explorarás cómo el marketing puede transformar ideas en experiencias memorables para los consumidores. Descubre cómo las marcas utilizan el storytelling y la innovación para conectarse con su audiencia.</p>
          <h3 className="text-lg font-semibold text-gray-800">Primera Lección: Introducción al Marketing</h3>
          <p className="text-gray-600">Bienvenido al mundo del marketing, donde la creatividad y la estrategia se encuentran. En esta lección, explorarás cómo el marketing puede transformar ideas en experiencias memorables para los consumidores. Descubre cómo las marcas utilizan el storytelling y la innovación para conectarse con su audiencia.</p>
          <div className="quiz-container bg-gray-100 p-4 rounded-lg shadow-md">
            <h4 className="text-md font-semibold text-gray-800">Ejercicio 1 de 2</h4>
            <p className="text-gray-600">¿Qué es el marketing?</p>
            <div className="space-y-2">
              <label className="flex items-center space-x-2">
                <input type="radio" name="marketing" value="Solo publicidad" onChange={() => handleAnswer("Solo publicidad", "Un conjunto de actividades para crear, comunicar y entregar valor a los clientes")} className="form-radio" />
                <span>Solo publicidad</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="radio" name="marketing" value="Un conjunto de actividades para crear, comunicar y entregar valor a los clientes" onChange={() => handleAnswer("Un conjunto de actividades para crear, comunicar y entregar valor a los clientes", "Un conjunto de actividades para crear, comunicar y entregar valor a los clientes")} className="form-radio" />
                <span>Un conjunto de actividades para crear, comunicar y entregar valor a los clientes</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="radio" name="marketing" value="Solo ventas" onChange={() => handleAnswer("Solo ventas", "Un conjunto de actividades para crear, comunicar y entregar valor a los clientes")} className="form-radio" />
                <span>Solo ventas</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="radio" name="marketing" value="Solo redes sociales" onChange={() => handleAnswer("Solo redes sociales", "Un conjunto de actividades para crear, comunicar y entregar valor a los clientes")} className="form-radio" />
                <span>Solo redes sociales</span>
              </label>
            </div>
            {feedback && <p className="mt-2 text-sm font-semibold">{feedback}</p>}
            <p className="mt-2 text-sm">Puntuación: {score} XP</p>
            <p className="mt-2 text-sm">Experiencia Global: {globalScore} XP</p>
          </div>
          <div className="quiz-container bg-gray-100 p-4 rounded-lg shadow-md">
            <h4 className="text-md font-semibold text-gray-800">Ejercicio 2 de 2</h4>
            <p className="text-gray-600">¿Cuál es el objetivo principal del marketing?</p>
            <div className="space-y-2">
              <label className="flex items-center space-x-2">
                <input type="radio" name="marketing-objective" value="Vender productos" onChange={() => handleAnswer("Vender productos", "Satisfacer las necesidades de los clientes")} className="form-radio" />
                <span>Vender productos</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="radio" name="marketing-objective" value="Satisfacer las necesidades de los clientes" onChange={() => handleAnswer("Satisfacer las necesidades de los clientes", "Satisfacer las necesidades de los clientes")} className="form-radio" />
                <span>Satisfacer las necesidades de los clientes</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="radio" name="marketing-objective" value="Aumentar el precio" onChange={() => handleAnswer("Aumentar el precio", "Satisfacer las necesidades de los clientes")} className="form-radio" />
                <span>Aumentar el precio</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="radio" name="marketing-objective" value="Ignorar la competencia" onChange={() => handleAnswer("Ignorar la competencia", "Satisfacer las necesidades de los clientes")} className="form-radio" />
                <span>Ignorar la competencia</span>
              </label>
            </div>
            {feedback && <p className="mt-2 text-sm font-semibold">{feedback}</p>}
            <p className="mt-2 text-sm">Puntuación: {score} XP</p>
            <p className="mt-2 text-sm">Experiencia Global: {globalScore} XP</p>
          </div>
          <h3 className="text-lg font-semibold text-gray-800">Guía de Marketing</h3>
          <p className="text-gray-600">Sumérgete en una guía completa que te llevará a través de las tendencias más emocionantes del marketing digital. Desde el uso de la inteligencia artificial para personalizar experiencias hasta estrategias de contenido que capturan la atención, esta guía te preparará para el futuro del marketing.</p>
          <h3 className="text-lg font-semibold text-gray-800">Cuestionarios</h3>
          <p className="text-gray-600">Completa cuestionarios para evaluar tu comprensión de los conceptos básicos.</p>
          <div className="quiz-container bg-gray-100 p-4 rounded-lg shadow-md">
            <h4 className="text-md font-semibold text-gray-800">Ejercicio 1 de 1</h4>
            <p className="text-gray-600">¿Qué es la segmentación de mercado?</p>
            <div className="space-y-2">
              <label className="flex items-center space-x-2">
                <input type="radio" name="market-segmentation" value="Dividir el mercado en grupos homogéneos" onChange={() => handleAnswer("Dividir el mercado en grupos homogéneos", "Dividir el mercado en grupos homogéneos")} className="form-radio" />
                <span>Dividir el mercado en grupos homogéneos</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="radio" name="market-segmentation" value="Ignorar las necesidades del cliente" onChange={() => handleAnswer("Ignorar las necesidades del cliente", "Dividir el mercado en grupos homogéneos")} className="form-radio" />
                <span>Ignorar las necesidades del cliente</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="radio" name="market-segmentation" value="Vender a todos los clientes" onChange={() => handleAnswer("Vender a todos los clientes", "Dividir el mercado en grupos homogéneos")} className="form-radio" />
                <span>Vender a todos los clientes</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="radio" name="market-segmentation" value="No realizar análisis de mercado" onChange={() => handleAnswer("No realizar análisis de mercado", "Dividir el mercado en grupos homogéneos")} className="form-radio" />
                <span>No realizar análisis de mercado</span>
              </label>
            </div>
            {feedback && <p className="mt-2 text-sm font-semibold">{feedback}</p>}
            <p className="mt-2 text-sm">Puntuación: {score} XP</p>
            <p className="mt-2 text-sm">Experiencia Global: {globalScore} XP</p>
            <button className="mt-4 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors">Siguiente Pregunta</button>
          </div>
          <div className="quiz-container bg-gray-100 p-4 rounded-lg shadow-md">
            <h4 className="text-md font-semibold text-gray-800">Ejercicio 2 de 2</h4>
            <p className="text-gray-600">¿Cuál es un beneficio de la segmentación de mercado?</p>
            <div className="space-y-2">
              <label className="flex items-center space-x-2">
                <input type="radio" name="market-segmentation-benefit" value="Aumentar la competencia" onChange={() => handleAnswer("Aumentar la competencia", "Mejorar la satisfacción del cliente")} className="form-radio" />
                <span>Aumentar la competencia</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="radio" name="market-segmentation-benefit" value="Mejorar la satisfacción del cliente" onChange={() => handleAnswer("Mejorar la satisfacción del cliente", "Mejorar la satisfacción del cliente")} className="form-radio" />
                <span>Mejorar la satisfacción del cliente</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="radio" name="market-segmentation-benefit" value="Ignorar las necesidades del cliente" onChange={() => handleAnswer("Ignorar las necesidades del cliente", "Mejorar la satisfacción del cliente")} className="form-radio" />
                <span>Ignorar las necesidades del cliente</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="radio" name="market-segmentation-benefit" value="No realizar análisis de mercado" onChange={() => handleAnswer("No realizar análisis de mercado", "Mejorar la satisfacción del cliente")} className="form-radio" />
                <span>No realizar análisis de mercado</span>
              </label>
            </div>
            {feedback && <p className="mt-2 text-sm font-semibold">{feedback}</p>}
            <p className="mt-2 text-sm">Puntuación: {score} XP</p>
            <p className="mt-2 text-sm">Experiencia Global: {globalScore} XP</p>
            <button className="mt-4 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors">Siguiente Pregunta</button>
          </div>
          <h3 className="text-lg font-semibold text-gray-800">Puzzles</h3>
          <p className="text-gray-600">Resuelve puzzles para aplicar lo que has aprendido de manera divertida.</p>
          <div className="puzzle-container bg-gray-100 p-4 rounded-lg shadow-md">
            <h4 className="text-md font-semibold text-gray-800">Puzzle: Ordena las palabras</h4>
            <p className="text-gray-600">Ordena las siguientes palabras para formar una frase relacionada con el marketing:</p>
            <ul className="list-disc pl-5 text-gray-600">
              <li>Marketing</li>
              <li>Digital</li>
              <li>Estrategia</li>
              <li>Contenido</li>
            </ul>
            <button className="mt-2 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors">Verificar</button>
          </div>
          <h3 className="text-lg font-semibold text-gray-800">Conceptos Básicos</h3>
          <p className="text-gray-600">Explora los conceptos fundamentales del marketing y su aplicación en el negocio.</p>
          <div className="quiz-container bg-gray-100 p-4 rounded-lg shadow-md">
            <h4 className="text-md font-semibold text-gray-800">Ejercicio 3 de 3</h4>
            <p className="text-gray-600">¿Cuál es una estrategia efectiva para el marketing digital?</p>
            <div className="space-y-2">
              <label className="flex items-center space-x-2">
                <input type="radio" name="digital-marketing" value="Ignorar las redes sociales" onChange={() => handleAnswer("Ignorar las redes sociales", "Utilizar SEO y contenido de calidad")} className="form-radio" />
                <span>Ignorar las redes sociales</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="radio" name="digital-marketing" value="Utilizar SEO y contenido de calidad" onChange={() => handleAnswer("Utilizar SEO y contenido de calidad", "Utilizar SEO y contenido de calidad")} className="form-radio" />
                <span>Utilizar SEO y contenido de calidad</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="radio" name="digital-marketing" value="No tener presencia en línea" onChange={() => handleAnswer("No tener presencia en línea", "Utilizar SEO y contenido de calidad")} className="form-radio" />
                <span>No tener presencia en línea</span>
              </label>
              <label className="flex items-center space-x-2">
                <input type="radio" name="digital-marketing" value="Solo usar publicidad pagada" onChange={() => handleAnswer("Solo usar publicidad pagada", "Utilizar SEO y contenido de calidad")} className="form-radio" />
                <span>Solo usar publicidad pagada</span>
              </label>
            </div>
            {feedback && <p className="mt-2 text-sm font-semibold">{feedback}</p>}
            <p className="mt-2 text-sm">Puntuación: {score} XP</p>
            <p className="mt-2 text-sm">Experiencia Global: {globalScore} XP</p>
          </div>
        </div>
        <button onClick={() => setSelectedPath(null)} className="text-blue-500 mt-4 hover:text-blue-600 transition-colors">Volver</button>
      </div>
    );
  }

  return (
    <motion.div 
      className="space-y-4 p-8 pt-6"
      variants={container}
      initial="hidden"
      animate="show"
    >
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {learningPaths.map((path) => {
          const Icon = Icons[path.icon];
          
          return (
            <motion.div key={path.id} variants={item}>
              <Card className="relative overflow-hidden cursor-pointer" onClick={() => handlePathClick(path)}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    {path.title}
                  </CardTitle>
                  <Icon className={cn("h-4 w-4", path.color)} />
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground mb-4">
                    {path.description}
                  </p>
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <Icons.bookOpen className="h-3 w-3 text-muted-foreground" />
                      <span>{path.lessons} lecciones</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Icons.star className="h-3 w-3 text-yellow-500" />
                      <span>{path.xp} XP</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          );
        })}
      </div>

      <motion.div variants={item} className="mt-8">
        <Card>
          <CardHeader>
            <CardTitle>Recomendaciones</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <Icons.target className="h-5 w-5 text-primary" />
                <div>
                  <h4 className="text-sm font-medium">Objetivos Diarios</h4>
                  <p className="text-xs text-muted-foreground">
                    Completa al menos 3 lecciones al día para mantener tu racha.
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <Icons.trophy className="h-5 w-5 text-yellow-500" />
                <div>
                  <h4 className="text-sm font-medium">Logros</h4>
                  <p className="text-xs text-muted-foreground">
                    Desbloquea logros especiales completando rutas de aprendizaje.
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <Icons.star className="h-5 w-5 text-blue-500" />
                <div>
                  <h4 className="text-sm font-medium">Experiencia</h4>
                  <p className="text-xs text-muted-foreground">
                    Gana XP completando lecciones y sube de nivel.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
}    