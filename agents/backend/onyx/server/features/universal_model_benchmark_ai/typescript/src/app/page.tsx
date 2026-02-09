'use client';

import { useState, useEffect } from 'react';
import BenchmarkDashboard from '@/components/BenchmarkDashboard';
import ModelSelector from '@/components/ModelSelector';
import ResultsTable from '@/components/ResultsTable';

export default function Home() {
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [selectedBenchmarks, setSelectedBenchmarks] = useState<string[]>([]);
  const [results, setResults] = useState<any[]>([]);
  const [isRunning, setIsRunning] = useState(false);

  const handleRunBenchmarks = async () => {
    setIsRunning(true);
    try {
      // Call API to run benchmarks
      const response = await fetch('/api/benchmarks/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          models: selectedModels,
          benchmarks: selectedBenchmarks,
        }),
      });
      
      const data = await response.json();
      setResults(data.results);
    } catch (error) {
      console.error('Error running benchmarks:', error);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-900 mb-8">
          Universal Model Benchmark AI
        </h1>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-2">
            <ModelSelector
              selected={selectedModels}
              onChange={setSelectedModels}
            />
          </div>
          
          <div>
            <BenchmarkSelector
              selected={selectedBenchmarks}
              onChange={setSelectedBenchmarks}
            />
          </div>
        </div>
        
        <div className="mb-8">
          <button
            onClick={handleRunBenchmarks}
            disabled={isRunning || selectedModels.length === 0 || selectedBenchmarks.length === 0}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {isRunning ? 'Running...' : 'Run Benchmarks'}
          </button>
        </div>
        
        {results.length > 0 && (
          <ResultsTable results={results} />
        )}
        
        <BenchmarkDashboard results={results} />
      </div>
    </main>
  );
}

function BenchmarkSelector({ selected, onChange }: { selected: string[], onChange: (value: string[]) => void }) {
  const benchmarks = ['mmlu', 'hellaswag', 'truthfulqa', 'gsm8k', 'humaneval'];
  
  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-4">Benchmarks</h2>
      <div className="space-y-2">
        {benchmarks.map((benchmark) => (
          <label key={benchmark} className="flex items-center">
            <input
              type="checkbox"
              checked={selected.includes(benchmark)}
              onChange={(e) => {
                if (e.target.checked) {
                  onChange([...selected, benchmark]);
                } else {
                  onChange(selected.filter(b => b !== benchmark));
                }
              }}
              className="mr-2"
            />
            <span className="capitalize">{benchmark}</span>
          </label>
        ))}
      </div>
    </div>
  );
}












