import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet, ScrollView, ActivityIndicator, Alert } from 'react-native';
import axios from 'axios';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { Picker } from '@react-native-picker/picker';

const API_BASE_URL = 'http://localhost:8000';

interface SmartContract {
  contract_id: string;
  name: string;
  address: string;
  network: string;
  abi: any[];
  bytecode: string;
  deployed_at: string;
}

interface NFT {
  nft_id: string;
  name: string;
  description: string;
  image_url: string;
  token_id: string;
  contract_address: string;
  owner: string;
  metadata: any;
}

interface DeFiProtocol {
  protocol_id: string;
  name: string;
  type: string;
  tvl: number;
  apy: number;
  risk_level: string;
  supported_tokens: string[];
}

const BlockchainScreen: React.FC = () => {
  const queryClient = useQueryClient();
  const [selectedNetwork, setSelectedNetwork] = useState<string>('ethereum');
  const [contractName, setContractName] = useState<string>('');
  const [contractCode, setContractCode] = useState<string>('');
  const [nftName, setNftName] = useState<string>('');
  const [nftDescription, setNftDescription] = useState<string>('');
  const [nftImageUrl, setNftImageUrl] = useState<string>('');

  // Deploy smart contract
  const deployContractMutation = useMutation<SmartContract, Error, any>(
    async (contractData: any) => {
      const response = await axios.post(`${API_BASE_URL}/api/blockchain/contracts`, contractData);
      return response.data;
    },
    {
      onSuccess: (data) => {
        Alert.alert('Success', `Smart contract deployed at: ${data.address}`);
        setContractName('');
        setContractCode('');
        queryClient.invalidateQueries('smartContracts');
      },
      onError: (error) => Alert.alert('Error', `Failed to deploy contract: ${error.message}`),
    }
  );

  // Create NFT
  const createNFTMutation = useMutation<NFT, Error, any>(
    async (nftData: any) => {
      const response = await axios.post(`${API_BASE_URL}/api/blockchain/nfts`, nftData);
      return response.data;
    },
    {
      onSuccess: (data) => {
        Alert.alert('Success', `NFT created with token ID: ${data.token_id}`);
        setNftName('');
        setNftDescription('');
        setNftImageUrl('');
        queryClient.invalidateQueries('nfts');
      },
      onError: (error) => Alert.alert('Error', `Failed to create NFT: ${error.message}`),
    }
  );

  // Get DeFi protocols
  const { data: defiProtocols, isLoading: isLoadingProtocols } = useQuery<DeFiProtocol[], Error>(
    'defiProtocols',
    async () => {
      const response = await axios.get(`${API_BASE_URL}/api/blockchain/defi/protocols`);
      return response.data;
    }
  );

  const handleDeployContract = () => {
    if (!contractName || !contractCode) {
      Alert.alert('Input Error', 'Contract name and code are required.');
      return;
    }
    deployContractMutation.mutate({
      name: contractName,
      code: contractCode,
      network: selectedNetwork
    });
  };

  const handleCreateNFT = () => {
    if (!nftName || !nftDescription || !nftImageUrl) {
      Alert.alert('Input Error', 'NFT name, description, and image URL are required.');
      return;
    }
    createNFTMutation.mutate({
      name: nftName,
      description: nftDescription,
      image_url: nftImageUrl,
      network: selectedNetwork
    });
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Blockchain & NFTs</Text>

      {/* Deploy Smart Contract */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Deploy Smart Contract</Text>
        <Text style={styles.label}>Contract Name:</Text>
        <TextInput style={styles.input} value={contractName} onChangeText={setContractName} placeholder="e.g., MyToken" />
        <Text style={styles.label}>Network:</Text>
        <Picker selectedValue={selectedNetwork} onValueChange={(itemValue) => setSelectedNetwork(itemValue)} style={styles.picker}>
          <Picker.Item label="Ethereum" value="ethereum" />
          <Picker.Item label="Polygon" value="polygon" />
          <Picker.Item label="BSC" value="bsc" />
          <Picker.Item label="Avalanche" value="avalanche" />
        </Picker>
        <Text style={styles.label}>Contract Code (Solidity):</Text>
        <TextInput
          style={[styles.input, styles.codeInput]}
          value={contractCode}
          onChangeText={setContractCode}
          placeholder="pragma solidity ^0.8.0;..."
          multiline
          numberOfLines={8}
        />
        <Button
          title={deployContractMutation.isLoading ? 'Deploying...' : 'Deploy Contract'}
          onPress={handleDeployContract}
          disabled={deployContractMutation.isLoading}
        />
        {deployContractMutation.isLoading && <ActivityIndicator style={styles.activityIndicator} />}
      </View>

      {/* Create NFT */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Create NFT</Text>
        <Text style={styles.label}>NFT Name:</Text>
        <TextInput style={styles.input} value={nftName} onChangeText={setNftName} placeholder="e.g., My Awesome NFT" />
        <Text style={styles.label}>Description:</Text>
        <TextInput style={styles.input} value={nftDescription} onChangeText={setNftDescription} placeholder="Describe your NFT..." multiline />
        <Text style={styles.label}>Image URL:</Text>
        <TextInput style={styles.input} value={nftImageUrl} onChangeText={setNftImageUrl} placeholder="https://example.com/image.png" />
        <Text style={styles.label}>Network:</Text>
        <Picker selectedValue={selectedNetwork} onValueChange={(itemValue) => setSelectedNetwork(itemValue)} style={styles.picker}>
          <Picker.Item label="Ethereum" value="ethereum" />
          <Picker.Item label="Polygon" value="polygon" />
          <Picker.Item label="BSC" value="bsc" />
          <Picker.Item label="Avalanche" value="avalanche" />
        </Picker>
        <Button
          title={createNFTMutation.isLoading ? 'Creating...' : 'Create NFT'}
          onPress={handleCreateNFT}
          disabled={createNFTMutation.isLoading}
        />
        {createNFTMutation.isLoading && <ActivityIndicator style={styles.activityIndicator} />}
      </View>

      {/* DeFi Protocols */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>DeFi Protocols</Text>
        {isLoadingProtocols ? (
          <ActivityIndicator />
        ) : (
          <View>
            {defiProtocols?.map((protocol) => (
              <View key={protocol.protocol_id} style={styles.protocolCard}>
                <Text style={styles.protocolName}>{protocol.name}</Text>
                <Text style={styles.protocolType}>Type: {protocol.type}</Text>
                <Text style={styles.protocolTVL}>TVL: ${protocol.tvl.toLocaleString()}</Text>
                <Text style={styles.protocolAPY}>APY: {protocol.apy}%</Text>
                <Text style={styles.protocolRisk}>Risk: {protocol.risk_level}</Text>
              </View>
            ))}
            {(!defiProtocols || defiProtocols.length === 0) && <Text>No DeFi protocols available.</Text>}
          </View>
        )}
      </View>

      {/* Blockchain Features */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Blockchain Features</Text>
        <View style={styles.featureList}>
          <Text style={styles.featureItem}>✓ Smart contract deployment</Text>
          <Text style={styles.featureItem}>✓ NFT creation and management</Text>
          <Text style={styles.featureItem}>✓ DeFi protocol integration</Text>
          <Text style={styles.featureItem}>✓ Multi-chain support</Text>
          <Text style={styles.featureItem}>✓ Web3 wallet integration</Text>
          <Text style={styles.featureItem}>✓ Token management</Text>
          <Text style={styles.featureItem}>✓ Decentralized identity</Text>
          <Text style={styles.featureItem}>✓ Content tokenization</Text>
          <Text style={styles.featureItem}>✓ Secure content provenance</Text>
          <Text style={styles.featureItem}>✓ Decentralized storage</Text>
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
    color: '#333',
  },
  section: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 15,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#444',
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
    marginTop: 10,
    marginBottom: 5,
    color: '#555',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 10,
    marginBottom: 15,
    backgroundColor: '#fff',
    fontSize: 16,
  },
  codeInput: {
    height: 120,
    textAlignVertical: 'top',
  },
  picker: {
    height: 50,
    width: '100%',
    marginBottom: 15,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    backgroundColor: '#fff',
  },
  activityIndicator: {
    marginTop: 10,
  },
  protocolCard: {
    backgroundColor: '#f9f9f9',
    borderRadius: 8,
    padding: 10,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#eee',
  },
  protocolName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  protocolType: {
    fontSize: 14,
    color: '#666',
  },
  protocolTVL: {
    fontSize: 14,
    color: '#666',
  },
  protocolAPY: {
    fontSize: 14,
    color: '#666',
  },
  protocolRisk: {
    fontSize: 14,
    color: '#666',
  },
  featureList: {
    marginTop: 10,
  },
  featureItem: {
    fontSize: 16,
    color: '#666',
    marginBottom: 8,
  },
});

export default BlockchainScreen;





















