import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet, ScrollView, ActivityIndicator, Alert, Switch } from 'react-native';
import axios from 'axios';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { Picker } from '@react-native-picker/picker';

const API_BASE_URL = 'http://localhost:8000';

interface SecurityPolicy {
  policy_id: string;
  name: string;
  description: string;
  framework: string;
  rules: any[];
  severity: string;
  is_active: boolean;
}

interface ComplianceReport {
  report_id: string;
  framework: string;
  generated_at: string;
  compliance_score: number;
  violations: string[];
  recommendations: string[];
}

const SecurityScreen: React.FC = () => {
  const queryClient = useQueryClient();
  const [selectedFramework, setSelectedFramework] = useState<string>('gdpr');
  const [newPolicyName, setNewPolicyName] = useState<string>('');
  const [newPolicyDescription, setNewPolicyDescription] = useState<string>('');

  // Generate compliance report
  const generateReportMutation = useMutation<ComplianceReport, Error, string>(
    async (framework: string) => {
      const response = await axios.post(`${API_BASE_URL}/api/security/compliance/reports`, { framework });
      return response.data;
    },
    {
      onSuccess: (data) => {
        Alert.alert('Success', `Compliance report generated. Score: ${data.compliance_score}%`);
        queryClient.invalidateQueries('complianceReports');
      },
      onError: (error) => Alert.alert('Error', `Failed to generate report: ${error.message}`),
    }
  );

  // Create security policy
  const createPolicyMutation = useMutation<SecurityPolicy, Error, any>(
    async (policyData: any) => {
      const response = await axios.post(`${API_BASE_URL}/api/security/policies`, policyData);
      return response.data;
    },
    {
      onSuccess: () => {
        Alert.alert('Success', 'Security policy created successfully!');
        setNewPolicyName('');
        setNewPolicyDescription('');
        queryClient.invalidateQueries('securityPolicies');
      },
      onError: (error) => Alert.alert('Error', `Failed to create policy: ${error.message}`),
    }
  );

  const handleGenerateReport = () => {
    generateReportMutation.mutate(selectedFramework);
  };

  const handleCreatePolicy = () => {
    if (!newPolicyName || !newPolicyDescription) {
      Alert.alert('Input Error', 'Policy name and description are required.');
      return;
    }
    createPolicyMutation.mutate({
      name: newPolicyName,
      description: newPolicyDescription,
      framework: selectedFramework,
      severity: 'high',
      rules: []
    });
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Advanced Security & Compliance</Text>

      {/* Generate Compliance Report */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Generate Compliance Report</Text>
        <Text style={styles.label}>Select Framework:</Text>
        <Picker selectedValue={selectedFramework} onValueChange={(itemValue) => setSelectedFramework(itemValue)} style={styles.picker}>
          <Picker.Item label="GDPR" value="gdpr" />
          <Picker.Item label="HIPAA" value="hipaa" />
          <Picker.Item label="PCI-DSS" value="pci_dss" />
          <Picker.Item label="SOX" value="sox" />
          <Picker.Item label="ISO27001" value="iso27001" />
          <Picker.Item label="NIST" value="nist" />
        </Picker>
        <Button
          title={generateReportMutation.isLoading ? 'Generating...' : 'Generate Report'}
          onPress={handleGenerateReport}
          disabled={generateReportMutation.isLoading}
        />
        {generateReportMutation.isLoading && <ActivityIndicator style={styles.activityIndicator} />}
      </View>

      {/* Create Security Policy */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Create Security Policy</Text>
        <Text style={styles.label}>Policy Name:</Text>
        <TextInput style={styles.input} value={newPolicyName} onChangeText={setNewPolicyName} placeholder="e.g., Data Encryption Policy" />
        <Text style={styles.label}>Description:</Text>
        <TextInput style={styles.input} value={newPolicyDescription} onChangeText={setNewPolicyDescription} placeholder="Describe the policy..." multiline />
        <Text style={styles.label}>Framework:</Text>
        <Picker selectedValue={selectedFramework} onValueChange={(itemValue) => setSelectedFramework(itemValue)} style={styles.picker}>
          <Picker.Item label="GDPR" value="gdpr" />
          <Picker.Item label="HIPAA" value="hipaa" />
          <Picker.Item label="PCI-DSS" value="pci_dss" />
          <Picker.Item label="SOX" value="sox" />
          <Picker.Item label="ISO27001" value="iso27001" />
          <Picker.Item label="NIST" value="nist" />
        </Picker>
        <Button
          title={createPolicyMutation.isLoading ? 'Creating...' : 'Create Policy'}
          onPress={handleCreatePolicy}
          disabled={createPolicyMutation.isLoading}
        />
        {createPolicyMutation.isLoading && <ActivityIndicator style={styles.activityIndicator} />}
      </View>

      {/* Security Features */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Security Features</Text>
        <View style={styles.featureList}>
          <Text style={styles.featureItem}>✓ End-to-end encryption</Text>
          <Text style={styles.featureItem}>✓ Multi-factor authentication</Text>
          <Text style={styles.featureItem}>✓ Role-based access control</Text>
          <Text style={styles.featureItem}>✓ Comprehensive audit trails</Text>
          <Text style={styles.featureItem}>✓ GDPR compliance</Text>
          <Text style={styles.featureItem}>✓ HIPAA compliance</Text>
          <Text style={styles.featureItem}>✓ PCI-DSS compliance</Text>
          <Text style={styles.featureItem}>✓ SOX compliance</Text>
          <Text style={styles.featureItem}>✓ ISO27001 compliance</Text>
          <Text style={styles.featureItem}>✓ NIST compliance</Text>
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
  featureList: {
    marginTop: 10,
  },
  featureItem: {
    fontSize: 16,
    color: '#666',
    marginBottom: 8,
  },
});

export default SecurityScreen;





















