"use client";
import React, { createContext, useContext, useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';

interface MetaMaskContextType {
  isConnected: boolean;
  address: string | null;
  connect: () => Promise<void>;
  disconnect: () => void;
  isMetaMaskInstalled: boolean;
}

const MetaMaskContext = createContext<MetaMaskContextType>({
  isConnected: false,
  address: null,
  connect: async () => {},
  disconnect: () => {},
  isMetaMaskInstalled: false,
});

export const useMetaMask = () => useContext(MetaMaskContext);

export const MetaMaskProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [address, setAddress] = useState<string | null>(null);
  const [isMetaMaskInstalled, setIsMetaMaskInstalled] = useState(false);

  // Verificar si MetaMask está instalado
  useEffect(() => {
    const checkMetaMask = () => {
      const isInstalled = typeof window !== 'undefined' && 
        typeof window.ethereum !== 'undefined' && 
        window.ethereum.isMetaMask;
      setIsMetaMaskInstalled(isInstalled);
    };

    checkMetaMask();
  }, []);

  // Verificar conexión existente
  useEffect(() => {
    const checkConnection = async () => {
      if (!isMetaMaskInstalled) return;

      try {
        // Usar window.ethereum directamente
        const accounts = await window.ethereum.request({ method: 'eth_accounts' });
        if (accounts.length > 0) {
          setAddress(accounts[0]);
          setIsConnected(true);
        }
      } catch (error) {
      }
    };

    checkConnection();

    // Escuchar cambios en la cuenta
    if (isMetaMaskInstalled && window.ethereum) {
      const handleAccountsChanged = (accounts: string[]) => {
        if (accounts.length === 0) {
          setAddress(null);
          setIsConnected(false);
        } else {
          setAddress(accounts[0]);
          setIsConnected(true);
        }
      };

      window.ethereum.on('accountsChanged', handleAccountsChanged);

      return () => {
        window.ethereum.removeListener('accountsChanged', handleAccountsChanged);
      };
    }
  }, [isMetaMaskInstalled]);

  const connect = async () => {
    if (!isMetaMaskInstalled) {
      toast.error('MetaMask no está instalado', {
        description: 'Por favor, instala MetaMask para continuar.',
        action: {
          label: 'Instalar MetaMask',
          onClick: () => window.open('https://metamask.io/download/', '_blank'),
        },
      });
      return;
    }

    try {
      // Usar window.ethereum directamente
      const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
      
      if (accounts.length > 0) {
        setAddress(accounts[0]);
        setIsConnected(true);
        toast.success('Conectado a MetaMask', {
          description: `Dirección: ${accounts[0].slice(0, 6)}...${accounts[0].slice(-4)}`,
        });
      }
    } catch (error) {
      if (error instanceof Error) {
        if (error.message.includes('User denied')) {
          toast.error('Conexión cancelada', {
            description: 'Has cancelado la conexión con MetaMask.',
          });
        } else {
          toast.error('Error al conectar con MetaMask', {
            description: 'Por favor, intenta de nuevo.',
          });
        }
      }
    }
  };

  const disconnect = () => {
    setAddress(null);
    setIsConnected(false);
    toast.info('Desconectado de MetaMask');
  };

  return (
    <MetaMaskContext.Provider value={{ isConnected, address, connect, disconnect, isMetaMaskInstalled }}>
      {children}
    </MetaMaskContext.Provider>
  );
};

export const ConnectWalletButton: React.FC = () => {
  const { isConnected, address, connect, disconnect, isMetaMaskInstalled } = useMetaMask();

  return (
    <Button
      variant={isConnected ? "outline" : "default"}
      onClick={isConnected ? disconnect : connect}
      className="flex items-center gap-2"
      disabled={!isMetaMaskInstalled}
    >
      {isConnected ? (
        <>
          <span className="w-2 h-2 bg-green-500 rounded-full" />
          {address?.slice(0, 6)}...{address?.slice(-4)}
        </>
      ) : (
        <>
          <span className="w-2 h-2 bg-red-500 rounded-full" />
          {isMetaMaskInstalled ? 'Conectar Wallet' : 'Instalar MetaMask'}
        </>
      )}
    </Button>
  );
};  