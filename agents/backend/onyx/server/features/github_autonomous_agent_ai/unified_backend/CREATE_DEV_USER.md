# Crear Usuario de Desarrollo

Este script crea un usuario de desarrollo para poder ver el frontend de Onyx.

## Credenciales de Desarrollo

- **Email:** `a@test.com`
- **Password:** `a`

## Métodos para Crear el Usuario

### Método 1: Script Python (Recomendado)

```bash
# Desde la raíz del proyecto
cd backend
python create_dev_user.py
```

Si tienes un venv activo:
```bash
source backend/.venv/bin/activate  # Linux/Mac
# o
backend\.venv\Scripts\activate    # Windows
python backend/create_dev_user.py
```

### Método 2: Script Bash

```bash
chmod +x backend/create_dev_user.sh
./backend/create_dev_user.sh
```

### Método 3: Usando curl directamente

```bash
curl -X POST http://localhost:3000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "a@test.com", "username": "a@test.com", "password": "a"}'
```

### Método 4: Desde el Frontend

1. Ve a `http://localhost:3000/auth/login`
2. Haz clic en "Don't have an account? Create one"
3. Crea una cuenta con cualquier email y contraseña

## Notas

- Asegúrate de que el backend esté corriendo antes de ejecutar el script
- Si el usuario ya existe, el script te informará y podrás usar las credenciales para login
- El script intenta conectarse a través del frontend proxy en `http://localhost:3000/api`
- Si el frontend no está corriendo, puedes cambiar la URL usando la variable de entorno `API_SERVER_URL`

## Verificar que el Backend está Corriendo

Puedes verificar los logs del backend:
```bash
# Ver logs del backend
tail -f backend/log/api_server_debug.log
```

O verificar que responde:
```bash
curl http://localhost:8080/health
```


















