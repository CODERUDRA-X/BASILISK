import React, { useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { Activity, AlertTriangle, DollarSign, Zap } from 'lucide-react';

function App() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  // Calls the FastAPI backend to run a full chaos simulation
  const runSimulation = async () => {
    setLoading(true);
    try {
      const response = await axios.get('http://localhost:8000/run_simulation');
      setResults(response.data);
    } catch (error) {
      console.error("Simulation failed:", error);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-8 font-mono">
      {/* Header Section */}
      <div className="max-w-6xl mx-auto mb-8 border-b border-gray-800 pb-4 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-emerald-400 tracking-wider">CODERUDRA-X</h1>
          <p className="text-gray-400 text-sm mt-1">Autonomous Chaos Trading Terminal v1.0</p>
        </div>
        <button 
          onClick={runSimulation}
          disabled={loading}
          className="bg-emerald-600 hover:bg-emerald-500 text-gray-900 font-bold py-3 px-6 rounded shadow-[0_0_15px_rgba(16,185,129,0.5)] transition-all disabled:opacity-50"
        >
          {loading ? 'SIMULATING CHAOS...' : 'DEPLOY AGENT'}
        </button>
      </div>

      {/* Metrics Dashboard */}
      {results && (
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div className="bg-gray-900 p-6 rounded-lg border border-gray-800 shadow-lg">
              <div className="flex items-center gap-3 mb-2"><Activity className="text-blue-400" /> <span className="text-gray-400">Total Steps</span></div>
              <div className="text-2xl font-bold">{results.total_steps}</div>
            </div>
            
            <div className="bg-gray-900 p-6 rounded-lg border border-gray-800 shadow-lg">
              <div className="flex items-center gap-3 mb-2"><DollarSign className="text-emerald-400" /> <span className="text-gray-400">Final Value</span></div>
              <div className={`text-2xl font-bold ${results.final_value > 1 ? 'text-emerald-400' : 'text-red-400'}`}>
                ${results.final_value.toFixed(4)}
              </div>
            </div>

            <div className="bg-gray-900 p-6 rounded-lg border border-gray-800 shadow-lg">
              <div className="flex items-center gap-3 mb-2"><Zap className="text-yellow-400" /> <span className="text-gray-400">Status</span></div>
              <div className="text-xl font-bold uppercase">{results.status.replace(/_/g, ' ')}</div>
            </div>

            <div className={`p-6 rounded-lg border shadow-lg ${results.status.includes('terminated') ? 'bg-red-900/20 border-red-500' : 'bg-gray-900 border-gray-800'}`}>
              <div className="flex items-center gap-3 mb-2">
                <AlertTriangle className={results.status.includes('terminated') ? 'text-red-500' : 'text-gray-400'} /> 
                <span className="text-gray-400">Risk Governor</span>
              </div>
              <div className="text-lg font-bold">
                {results.status.includes('terminated') ? 'MARGIN CALL (10% DD)' : 'SURVIVED'}
              </div>
            </div>
          </div>

          {/* Live Charting Section */}
          <div className="bg-gray-900 p-6 rounded-lg border border-gray-800 shadow-lg">
            <h2 className="text-xl font-bold mb-6 text-gray-300">Portfolio Value Trajectory</h2>
            <div className="h-96 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={results.chart_data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="step" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" domain={['auto', 'auto']} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#111827', borderColor: '#374151' }}
                    itemStyle={{ color: '#10B981' }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="portfolio_value" 
                    stroke="#10B981" 
                    strokeWidth={2} 
                    dot={false} 
                    activeDot={{ r: 8 }} 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;