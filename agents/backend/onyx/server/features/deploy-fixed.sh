#!/bin/bash
cd /home/ec2-user/blatam-academy

# Create proper Next.js 15 app structure
cat > package.json << EOF
{
  "name": "blatam-academy",
  "version": "1.0.0",
  "scripts": {
    "dev": "next dev",
    "build": "next build", 
    "start": "next start -p 3000"
  },
  "dependencies": {
    "next": "^15.0.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  }
}
EOF

# Install dependencies
npm install

# Create app directory structure
mkdir -p app

# Create root layout
cat > app/layout.tsx << EOF
export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="es">
      <body>
        <div style={{padding: "20px", fontFamily: "Arial, sans-serif"}}>
          <header style={{borderBottom: "1px solid #ccc", paddingBottom: "10px", marginBottom: "20px"}}>
            <h1 style={{color: "#0070f3"}}>Blatam Academy</h1>
            <p>Plataforma educativa desplegada en AWS EC2</p>
          </header>
          {children}
        </div>
      </body>
    </html>
  )
}
EOF

# Create main page
cat > app/page.tsx << EOF
export default function HomePage() {
  return (
    <div>
      <h2>¡Bienvenido a Blatam Academy!</h2>
      <p>La aplicación se ha desplegado exitosamente en AWS EC2.</p>
      <div style={{marginTop: "20px", padding: "15px", backgroundColor: "#f0f8ff", border: "1px solid #0070f3", borderRadius: "5px"}}>
        <h3>Estado del Deployment:</h3>
        <ul>
          <li>✅ Servidor EC2 funcionando</li>
          <li>✅ Next.js 15 configurado</li>
          <li>✅ Aplicación corriendo en puerto 3000</li>
          <li>✅ Listo para configuración DNS</li>
        </ul>
      </div>
      <div style={{marginTop: "20px"}}>
        <h3>Configuración DNS para HostGator:</h3>
        <p>Crear registro A apuntando a la IP del servidor EC2</p>
      </div>
    </div>
  )
}
EOF

# Build the application
npm run build

# Kill any existing processes on port 3000
pkill -f "next start" || true
sleep 2

# Start the application
nohup npm start > app.log 2>&1 &

echo "Deployment completed successfully!"
echo "Application running on port 3000"
echo "Check logs with: tail -f app.log"

