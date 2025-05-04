import { useState } from 'react';

export default function Home() {
  const [income, setIncome] = useState('');
  const [expenses, setExpenses] = useState('');
  const [creditScore, setCreditScore] = useState('');
  const [employmentStatus, setEmploymentStatus] = useState('');
  const [eligibility, setEligibility] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Sending form data to API for eligibility check
    const response = await fetch('/api/checkEligibility', {
      method: 'POST',
      body: JSON.stringify({
        income: parseFloat(income),
        expenses: parseFloat(expenses),
        creditScore: parseInt(creditScore),
        employmentStatus,
      }),
      headers: {
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();
    setEligibility(data.eligibility);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="p-8 max-w-lg w-full bg-white shadow-lg rounded-xl">
        <h1 className="text-2xl font-bold mb-4">Loan Application</h1>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-sm font-semibold mb-2">Income (R)</label>
            <input
              type="number"
              className="w-full p-2 border rounded"
              value={income}
              onChange={(e) => setIncome(e.target.value)}
              required
            />
          </div>
          <div className="mb-4">
            <label className="block text-sm font-semibold mb-2">Expenses (R)</label>
            <input
              type="number"
              className="w-full p-2 border rounded"
              value={expenses}
              onChange={(e) => setExpenses(e.target.value)}
              required
            />
          </div>
          <div className="mb-4">
            <label className="block text-sm font-semibold mb-2">Credit Score</label>
            <input
              type="number"
              className="w-full p-2 border rounded"
              value={creditScore}
              onChange={(e) => setCreditScore(e.target.value)}
              required
            />
          </div>
          <div className="mb-4">
            <label className="block text-sm font-semibold mb-2">Employment Status</label>
            <select
              className="w-full p-2 border rounded"
              value={employmentStatus}
              onChange={(e) => setEmploymentStatus(e.target.value)}
              required
            >
              <option value="">Select</option>
              <option value="Employed">Employed</option>
              <option value="Self-Employed">Self-Employed</option>
              <option value="Unemployed">Unemployed</option>
            </select>
          </div>
          <button type="submit" className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700">
            Check Eligibility
          </button>
        </form>

        {eligibility && (
          <div className="mt-4 p-4 bg-green-100 rounded-md">
            <p>Your loan eligibility status: <strong>{eligibility}</strong></p>
          </div>
        )}
      </div>
    </div>
  );
}
