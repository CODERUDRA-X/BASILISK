import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { ShieldAlert, Zap, DollarSign, Activity } from 'lucide-react';

function App() {
  const [data, setData] = useState([]);
  const [status, setStatus] = useState('STANDBY');
  const [shock, setShock] = useState(null);
  const ws = useRef(null);

  const startSimulation = () => {
    setData([]);
    setStatus('INGESTING LIVE CHAOS...');
    ws.current = new WebSocket('ws://localhost:8000/ws/simulate');

    ws.current.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'STEP') {
        setData(prev => [...prev, msg]);
        if (msg.status === 'TERMINATED') {
          setStatus('MARGIN CALL - SYSTEM HALTED');
          ws.current.close();
        }
      } else if (msg.type === 'SHOCK') {
        setStatus('AWAITING HUMAN OVERRIDE');
        setShock(msg);
      }
    };
  };

  const handleAuthorize = () => {
    setShock(null);
    setStatus('CHAOS FROZEN. EXECUTING HEDGE...');
    ws.current.send('AUTHORIZE');
  };

  return (
    <div className={`min-h-screen text-gray-100 p-8 font-mono transition-colors duration-500 ${shock ? 'bg-red-950' : 'bg-gray-950'}`}>
      
      {/* Header */}
      <div className="max-w-6xl mx-auto mb-8 border-b border-gray-800 pb-4 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-emerald-400 tracking-wider">PROJECT BASILISK</h1>
          <p className="text-gray-400 text-sm mt-1">Autonomous Chaos Trading Terminal v1.0</p>
        </div>
        <button 
          onClick={startSimulation}
          disabled={status !== 'STANDBY'}
          className="bg-emerald-600 hover:bg-emerald-500 text-gray-900 font-bold py-3 px-6 rounded shadow-[0_0_15px_rgba(16,185,129,0.5)] disabled:opacity-50"
        >
          {status === 'STANDBY' ? 'DEPLOY AGENT' : 'SYSTEM ONLINE'}
        </button>
      </div>

      {/* Shock Modal */}
      {shock && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
          <div className="bg-red-950 border-2 border-red-500 p-8 rounded-lg max-w-2xl text-center shadow-[0_0_50px_rgba(239,68,68,0.5)]">
            <ShieldAlert className="w-20 h-20 text-red-500 mx-auto mb-4 animate-pulse" />
            <h2 className="text-3xl font-bold text-red-500 mb-2">🚨 CRITICAL MARKET SHOCK DETECTED</h2>
            <p className="text-gray-300 mb-6 text-lg">
              Live Data Anomaly Variance: <span className="font-bold text-white">{shock.variance}</span><br/>
              Basilisk predicts severe drawdown. Awaiting Human Authorization to execute defensive hedge.
            </p>
            <div className="flex gap-4 justify-center">
              <button 
                onClick={handleAuthorize}
                className="bg-red-600 hover:bg-red-500 text-white font-bold py-3 px-8 rounded shadow-[0_0_15px_rgba(239,68,68,0.8)] uppercase"
              >
                [ AUTHORIZE HEDGE ]
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Dashboard */}
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="bg-gray-900 p-6 rounded-lg border border-gray-800 shadow-lg">
              <div className="flex items-center gap-3 mb-2"><Activity className="text-blue-400" /> <span className="text-gray-400">Total Steps</span></div>
              <div className="text-2xl font-bold">{data.length}</div>
            </div>
            <div className="bg-gray-900 p-6 rounded-lg border border-gray-800 shadow-lg">
              <div className="flex items-center gap-3 mb-2"><DollarSign className="text-emerald-400" /> <span className="text-gray-400">Portfolio Value</span></div>
              <div className="text-2xl font-bold text-emerald-400">
                ${data.length > 0 ? data[data.length - 1].portfolio_value.toFixed(4) : '1.0000'}
              </div>
            </div>
            <div className="bg-gray-900 p-6 rounded-lg border border-gray-800 shadow-lg">
              <div className="flex items-center gap-3 mb-2"><Zap className="text-yellow-400" /> <span className="text-gray-400">Status</span></div>
              <div className="text-lg font-bold text-yellow-400 uppercase animate-pulse">{status}</div>
            </div>
        </div>

        <div className="bg-gray-900 p-6 rounded-lg border border-gray-800 shadow-lg relative">
          <h2 className="text-xl font-bold mb-6 text-gray-300">Live Attention Trajectory</h2>
          <div className="h-96 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="step" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" domain={['auto', 'auto']} />
                <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151' }} itemStyle={{ color: '#10B981' }} />
                <Line type="stepAfter" dataKey="portfolio_value" stroke={shock ? '#EF4444' : '#10B981'} strokeWidth={3} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;