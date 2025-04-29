import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { CardNumberElement, CardExpiryElement, CardCvcElement, useStripe, useElements } from '@stripe/react-stripe-js';

// Pricing tiers
const prices = {
  free: { label: 'Tier 1 ', amount: 0 },
  tier2: { label: 'Tier 2', amount: 2000 },
  tier3: { label: 'Tier 3', amount: 3000 },
  tier4: { label: 'Tier 4', amount: 10000 },
};

const elementStyles = {
  style: {
    base: {
      fontSize: '16px',
      color: '#7dc4ff', 
      fontFamily: '"Fira Sans", sans-serif',
      '::placeholder': {
        color: '#7dc4ff',
      },
    },
    invalid: {
      color: '#ff6b6b',
    },
  },
};

const CheckoutForm = () => {
  const navigate = useNavigate();
  const { plan } = useParams();
  const stripe = useStripe();
  const elements = useElements();
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [success, setSuccess] = useState(false);
  const [selectedPlan, setSelectedPlan] = useState(prices.free); // Default to free tier

  useEffect(() => {
    // Validate the plan parameter and set the selected plan
    if (plan && prices[plan]) {
      setSelectedPlan(prices[plan]);
    } else {
      setSelectedPlan(prices.free);
      setMessage('Invalid plan selected. Defaulting to free tier.');
    }
  }, [plan]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedPlan) return;
    
    // Prevent submission for free tier
    if (selectedPlan.amount === 0) {
      setMessage('✅ Payment successful! Redirecting...');
      setSuccess(true);
      setTimeout(() => navigate('/'), 3000);
      return;
    }
    
    setLoading(true);

    const card = elements.getElement(CardNumberElement);

    const { error, paymentMethod } = await stripe.createPaymentMethod({
      type: 'card',
      card,
    });

    if (error) {
      setMessage(error.message);
      setLoading(false);
      return;
    }

    try {
      const res = await fetch('http://localhost:5050/pay', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          payment_method: paymentMethod.id,
          amount: selectedPlan.amount,
        }),
      });

      const data = await res.json();
      if (data.success) {
        setMessage('✅ Payment successful! Redirecting...');
        setSuccess(true);
        setTimeout(() => navigate('/'), 5000);
      } else {
        setMessage(`❌ ${data.error || 'Payment failed'}`);
      }
    } catch (err) {
      setMessage('❌ Failed to connect to the server.');
    }

    setLoading(false);
  };

  return (
    <div className="w-full min-h-screen flex justify-center items-center bg-gradient-to-r from-purple-600 to-blue-500 px-4">
      <div className="w-full max-w-md bg-[#0f172a] text-white p-8 rounded-2xl shadow-2xl space-y-6">
        <h2 className="text-2xl font-bold text-center">Pay for {selectedPlan?.label}</h2>

        {!success && (
          <>
            <form onSubmit={handleSubmit}>
              <div>
                <label className="form-label block font-bold text-[#dcd0ff] mb-2">Card Number:</label>
                <div className="rounded-md overflow-hidden border border-[#7dc4ff] bg-[#2e1b4f] px-4 py-2">
                  <CardNumberElement options={elementStyles} className="w-full" />
                </div>
              </div>

              <div className="flex gap-4">
                <div className="w-1/2">
                  <label className="form-label block font-bold text-[#dcd0ff] mb-2">Expiration:</label>
                  <div className="rounded-md overflow-hidden border border-[#7dc4ff] bg-[#2e1b4f] px-4 py-2">
                    <CardExpiryElement options={elementStyles} className="w-full" />
                  </div>
                </div>
                <div className="w-1/2">
                  <label className="form-label block font-bold text-[#dcd0ff] mb-2">CVC:</label>
                  <div className="rounded-md overflow-hidden border border-[#7dc4ff] bg-[#2e1b4f] px-4 py-2">
                    <CardCvcElement options={elementStyles} className="w-full" />
                  </div>
                </div>
              </div>

              <button
                type="submit"
                disabled={!stripe || loading}
                className="w-full bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white py-2 px-4 rounded-md shadow-md transition duration-300 font-bold mt-4"
              >
                {loading ? 'Processing...' : `Pay $${(selectedPlan.amount / 100).toFixed(2)}`}
              </button>
            </form>
          </>
        )}

        {message && (
          <div className="mt-2 text-center text-sm text-purple-200">
            {message}
          </div>
        )}
      </div>
    </div>
  );
};

export default CheckoutForm;
