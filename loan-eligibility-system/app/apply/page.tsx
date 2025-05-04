'use client';

import { useState } from 'react';
import { supabase } from '@/src/lib/supabaseClient';

export default function LoanApplicationPage() {
  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    income: '',
    expenses: '',
    creditScore: '',
    employmentStatus: '',
  });

  const [message, setMessage] = useState('');
  const [eligibility, setEligibility] = useState('');
  const [interestRate, setInterestRate] = useState<number | null>(null);

  // Eligibility logic
  const checkEligibility = () => {
    const disposableIncome = parseFloat(formData.income) - parseFloat(formData.expenses);
    const isEligible =
      disposableIncome > 3000 &&
      parseInt(formData.creditScore) >= 650 &&
      (formData.employmentStatus === 'employed' || formData.employmentStatus === 'self-employed');

    if (!isEligible) {
      setEligibility('Sorry, you are not eligible for a loan.');
      setInterestRate(null); // No interest rate for ineligible users
    } else {
      setEligibility('You are eligible for a loan!');
      calculateInterestRate(parseInt(formData.creditScore));
    }
  };

  // Interest rate calculation based on credit score
  const calculateInterestRate = (creditScore: number) => {
    if (creditScore >= 800) {
      setInterestRate(6); // 6% interest rate for credit score 800+
    } else if (creditScore >= 751) {
      setInterestRate(8); // 8% interest rate for credit score 751–800
    } else if (creditScore >= 701) {
      setInterestRate(10); // 10% interest rate for credit score 701–750
    } else if (creditScore >= 650) {
      setInterestRate(12); // 12% interest rate for credit score 650–700
    } else {
      setInterestRate(null); // No interest rate for credit score below 650
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Check eligibility before submitting
    checkEligibility();

    // If eligible, submit to Supabase
    if (eligibility === 'You are eligible for a loan!') {
      const { error } = await supabase.from('applications').insert([
        {
          full_name: formData.fullName,
          email: formData.email,
          income: parseFloat(formData.income),
          expenses: parseFloat(formData.expenses),
          credit_score: parseInt(formData.creditScore),
          employment_status: formData.employmentStatus,
          interest_rate: interestRate, // Save the calculated interest rate
        },
      ]);

      if (error) {
        console.error('Submission error:', error.message);
        setMessage('Failed to submit. Please try again.');
      } else {
        setMessage('Application submitted successfully!');
        setFormData({
          fullName: '',
          email: '',
          income: '',
          expenses: '',
          creditScore: '',
          employmentStatus: '',
        });
        setInterestRate(null);
      }
    }
  };

  return (
    <main className="max-w-xl mx-auto py-12 px-4">
      <h1 className="text-2xl font-bold mb-6">Loan Application Form</h1>
      <form onSubmit={handleSubmit} className="space-y-4 bg-white shadow p-6 rounded-lg">
        <input
          name="fullName"
          placeholder="Full Name"
          value={formData.fullName}
          onChange={handleChange}
          className="w-full border px-3 py-2 rounded"
          required
        />
        <input
          type="email"
          name="email"
          placeholder="Email Address"
          value={formData.email}
          onChange={handleChange}
          className="w-full border px-3 py-2 rounded"
          required
        />
        <input
          type="number"
          name="income"
          placeholder="Monthly Income (ZAR)"
          value={formData.income}
          onChange={handleChange}
          className="w-full border px-3 py-2 rounded"
          required
        />
        <input
          type="number"
          name="expenses"
          placeholder="Monthly Expenses (ZAR)"
          value={formData.expenses}
          onChange={handleChange}
          className="w-full border px-3 py-2 rounded"
          required
        />
        <input
          type="number"
          name="creditScore"
          placeholder="Credit Score"
          value={formData.creditScore}
          onChange={handleChange}
          className="w-full border px-3 py-2 rounded"
          required
        />
        <select
          name="employmentStatus"
          value={formData.employmentStatus}
          onChange={handleChange}
          className="w-full border px-3 py-2 rounded"
          required
        >
          <option value="">Employment Status</option>
          <option value="employed">Employed</option>
          <option value="self-employed">Self-employed</option>
          <option value="unemployed">Unemployed</option>
          <option value="student">Student</option>
        </select>
        <button
          type="submit"
          className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700"
        >
          Submit Application
        </button>
      </form>

      {/* Eligibility message */}
      {eligibility && <p className="mt-4 text-center text-sm">{eligibility}</p>}

      {/* Display the interest rate if eligible */}
      {eligibility === 'You are eligible for a loan!' && interestRate && (
        <p className="mt-4 text-center text-sm">
          Congratulations! Your interest rate is {interestRate}%.
        </p>
      )}

      {/* Submission result */}
      {message && <p className="mt-4 text-center text-sm">{message}</p>}
    </main>
  );
}
