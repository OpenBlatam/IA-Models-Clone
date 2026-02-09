#!/bin/bash
# Script de setup para desarrollo - Robot Movement AI v2.0

set -e

echo "🚀 Configurando entorno de desarrollo para Robot Movement AI v2.0..."

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar Python
echo -e "${BLUE}📦 Verificando Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no encontrado. Por favor instala Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✅ Python ${PYTHON_VERSION} encontrado${NC}"

# Crear entorno virtual
echo -e "${BLUE}🔧 Creando entorno virtual...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Entorno virtual creado${NC}"
else
    echo -e "${YELLOW}⚠️  Entorno virtual ya existe${NC}"
fi

# Activar entorno virtual
echo -e "${BLUE}🔌 Activando entorno virtual...${NC}"
source venv/bin/activate

# Actualizar pip
echo -e "${BLUE}📥 Actualizando pip...${NC}"
pip install --upgrade pip setuptools wheel

# Instalar dependencias
echo -e "${BLUE}📦 Instalando dependencias...${NC}"
pip install -r requirements.txt

# Instalar dependencias de desarrollo
echo -e "${BLUE}🔧 Instalando dependencias de desarrollo...${NC}"
pip install pytest pytest-asyncio pytest-cov black flake8 mypy

# Crear directorios necesarios
echo -e "${BLUE}📁 Creando directorios...${NC}"
mkdir -p db logs tests/coverage

# Configurar archivo .env si no existe
if [ ! -f ".env" ]; then
    echo -e "${BLUE}⚙️  Creando archivo .env...${NC}"
    cat > .env << EOF
# Robot Configuration
ROBOT_IP=192.168.1.100
ROBOT_PORT=30001
ROBOT_BRAND=kuka
ROS_ENABLED=false
FEEDBACK_FREQUENCY=1000

# LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here

# Database Configuration
DATABASE_URL=sqlite:///./db/robots.db
REPOSITORY_TYPE=in_memory
CACHE_TTL=300

# API Configuration
API_HOST=0.0.0.0
API_PORT=8010

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/robot_movement_ai.log

# Development
DEBUG=true
EOF
    echo -e "${GREEN}✅ Archivo .env creado${NC}"
    echo -e "${YELLOW}⚠️  Por favor edita .env con tus configuraciones${NC}"
else
    echo -e "${YELLOW}⚠️  Archivo .env ya existe${NC}"
fi

# Ejecutar tests
echo -e "${BLUE}🧪 Ejecutando tests...${NC}"
pytest tests/ -v --cov=core --cov-report=html --cov-report=term || echo -e "${YELLOW}⚠️  Algunos tests fallaron${NC}"

# Verificar imports
echo -e "${BLUE}🔍 Verificando imports...${NC}"
python -c "import robot_movement_ai; print('✅ Imports OK')" || echo -e "${YELLOW}⚠️  Problemas con imports${NC}"

echo ""
echo -e "${GREEN}✅ Setup completado exitosamente!${NC}"
echo ""
echo -e "${BLUE}📝 Próximos pasos:${NC}"
echo -e "  1. Edita el archivo .env con tus configuraciones"
echo -e "  2. Activa el entorno virtual: source venv/bin/activate"
echo -e "  3. Ejecuta la aplicación: python -m robot_movement_ai.main"
echo -e "  4. Lee START_HERE.md para más información"
echo ""




