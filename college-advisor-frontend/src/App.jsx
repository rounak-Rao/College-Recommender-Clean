import React, { useState } from 'react';
import './App.css'; // optional for further styling

function CollegeRecommender() {
  const [schools, setSchools] = useState('');
  const [income, setIncome] = useState('');
  const [major, setMajor] = useState('');
  const [state, setState] = useState('');
  const [parentLoan, setParentLoan] = useState(false);
  const [preference, setPreference] = useState(50);
  const [result, setResult] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();

    const data = {
      colleges: schools.split(',').map(s => s.trim()),
      major,
      income_tier: parseInt(income),
      user_state:state,
      parent_loans: parentLoan,
      weight_qol: (100 - preference) / 100,
      weight_roi: preference / 100
    };
    console.log("Payload being sent to backend:", data);

    try {
      const response = await fetch('http://localhost:8000/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });

      const resData = await response.json();
      setResult(resData.explanation || 'No recommendation returned.');
    } catch (error) {
      console.error('Error fetching recommendation:', error);
      setResult("Something went wrong. Please try again.");
    }
  };

  return (
    <div style={{
      maxWidth: '700px',
      margin: '0 auto',
      padding: '2rem',
      fontFamily: 'Segoe UI, sans-serif',
      lineHeight: 1.6
    }}>
      <h1 style={{ textAlign: 'center', marginBottom: '0.5rem' }}>Smart College Recommender</h1>
      <p style={{ textAlign: 'center', marginBottom: '2rem', fontSize: '1.05rem' }}>
        This tool helps you choose the best college based on your goals. Just enter your preferences below — including income, intended major,
        and whether you prioritize Return on Investment (ROI) or Quality of Life (QoL). We’ll analyze everything and recommend the best fit.
      </p>

      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
        <label>
          <strong>List of Colleges (comma-separated):</strong>
          <textarea
            rows={3}
            placeholder="e.g., UC Irvine, Stanford University, UCLA"
            style={{ width: '100%', padding: '0.5rem', resize: 'vertical' }}
            value={schools}
            onChange={e => setSchools(e.target.value)}
            required
          />
        </label>

        <label>
          <strong>Household Income:</strong>
          <select
            value={income}
            onChange={e => setIncome(e.target.value)}
            required
            style={{ padding: '0.5rem', width: '100%' }}
          >
            <option value="">Select income range...</option>
            <option value="0">Below $30,000</option>
            <option value="1">$30,001 – $75,000</option>
            <option value="2">Above $75,000</option>
          </select>
        </label>

        <label>
          <strong>Intended Major:</strong>
          <input
            type="text"
            value={major}
            onChange={e => setMajor(e.target.value)}
            placeholder="e.g., Computer Science"
            style={{ width: '100%', padding: '0.5rem' }}
            required
          />
        </label>

        <label>
          <strong>State of Residence (e.g., CA, NY):</strong>
          <input
            type="text"
            value={state}
            onChange={e => setState(e.target.value)}
            placeholder="e.g., CA"
            style={{ width: '100%', padding: '0.5rem' }}
            required
          />
        </label>

        <label>
          <strong>Will you take out Parent PLUS Loans?</strong>
          <input
            type="checkbox"
            checked={parentLoan}
            onChange={e => setParentLoan(e.target.checked)}
            style={{ marginLeft: '0.5rem' }}
          />
        </label>

        <label>
          <strong>Preference Slider: Quality of Life vs ROI</strong>
          <input
            type="range"
            min="0"
            max="100"
            value={preference}
            onChange={e => setPreference(e.target.value)}
            style={{ width: '100%' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem' }}>
            <span>100% QoL</span>
            <span>50/50</span>
            <span>100% ROI</span>
          </div>
        </label>

        <button
          type="submit"
          style={{
            padding: '0.75rem',
            fontSize: '1rem',
            backgroundColor: '#3b82f6',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer'
          }}
        >
          Get My Recommendation
        </button>
      </form>

      {result && (
      <div style={{
        marginTop: '2rem',
        backgroundColor: '#ecf4ff',
        padding: '1.5rem',
        borderRadius: '8px',
        border: '1px solid #d0e3ff',
        boxShadow: '0 2px 8px rgba(0,0,0,0.08)'
      }}>
        <h2 style={{ color: '#2563eb', marginBottom: '0.75rem' }}>🎓 Recommendation</h2>
        <p style={{ fontSize: '1.05rem', lineHeight: 1.6, color: '#111827' }}>{result}</p>
      </div>
    )}
    </div>
  );
}

export default CollegeRecommender;
