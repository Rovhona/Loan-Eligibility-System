import type { NextApiRequest, NextApiResponse } from 'next';

type Data = {
  eligibility: string;
};

export default function handler(req: NextApiRequest, res: NextApiResponse<Data>) {
  if (req.method === 'POST') {
    const { income, expenses, creditScore, employmentStatus } = req.body;

    // Eligibility check logic
    if (creditScore < 550) {
      return res.status(200).json({ eligibility: 'Rejected (Low credit score)' });
    }
    if (income - expenses < 5000) {
      return res.status(200).json({ eligibility: 'Rejected (Low disposable income)' });
    }
    if (employmentStatus === 'Unemployed') {
      return res.status(200).json({ eligibility: 'Rejected (Unemployed)' });
    }

    return res.status(200).json({ eligibility: 'Approved' });
  } else {
    res.status(405).json({ eligibility: 'Method Not Allowed' });
  }
}
//               onChange={(e) => setCreditScore(e.target.value)}
//               required       
//             />
//           </div>
//           <div className="mb-4">
//             <label className="block text-sm font-semibold mb-2">Employment Status</label>
//             <select
    