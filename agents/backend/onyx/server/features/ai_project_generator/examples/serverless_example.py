"""
Ejemplo Serverless - Configuración para serverless
==================================================

Ejemplo optimizado para AWS Lambda, Azure Functions, etc.
"""

from core.easy_setup import create_app_serverless
from mangum import Mangum

# Crear aplicación serverless
app = create_app_serverless()

# Crear handler para Lambda
handler = Mangum(app, lifespan="off")

# Para usar en Lambda:
# handler = Mangum(app)

# Para testing local:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)















