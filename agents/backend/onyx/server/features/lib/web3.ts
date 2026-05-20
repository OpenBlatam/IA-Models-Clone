import { ethers } from 'ethers';

export const getProvider = () => {
  if (typeof window === 'undefined') return null;
  
  try {
    if (window.ethereum) {
      return new ethers.providers.Web3Provider(window.ethereum);
    } else {
      return null;
    }
  } catch (error) {
    return null;
  }
};

export const connectWallet = async () => {
  if (typeof window === 'undefined') {
    throw new Error('MetaMask only available in browser environment');
  }
  
  try {
    if (!window.ethereum) {
      throw new Error('MetaMask no está instalado. Por favor, instala MetaMask para continuar.');
    }

    const provider = new ethers.providers.Web3Provider(window.ethereum);
    await provider.send("eth_requestAccounts", []);
    const signer = provider.getSigner();
    const address = await signer.getAddress();
    return { address, provider, signer };
  } catch (error) {
    if (error instanceof Error) {
      if (error.message.includes('MetaMask no está instalado')) {
        window.open('https://metamask.io/download/', '_blank');
      }
      throw error;
    }
    throw new Error('Error al conectar con la wallet');
  }
};

export const getBalance = async (address: string) => {
  try {
    const provider = getProvider();
    if (!provider) {
      throw new Error('No se pudo conectar con MetaMask');
    }
    const balance = await provider.getBalance(address);
    return ethers.utils.formatEther(balance);
  } catch (error) {
    return '0';
  }
};

// Tipos para TypeScript
declare global {
  interface Window {
    ethereum?: any;
  }
}    