import React, { useState } from 'react';
import { db } from '../firebase';
import { collection, addDoc, Timestamp } from 'firebase/firestore';

export const Newsletter = () => {
  const [email, setEmail] = useState('');
  const [status, setStatus] = useState('');
  const [isError, setIsError] = useState(false);

  const isValidEmail = (email) => {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return regex.test(email);
  };

  const handleSubscribe = async () => {
    if (!email) {
      setStatus("Please enter your email!");
      setIsError(true);
      return;
    }

    if (!isValidEmail(email)) {
      setStatus("Invalid email format. Please try again.");
      setIsError(true);
      return;
    }

    try {
      await addDoc(collection(db, "subscribers"), {
        email: email,
        subscribedAt: Timestamp.now()
      });
      setStatus("Subscribed successfully!");
      setIsError(false);
      setEmail('');
    } catch (error) {
      console.error("Error subscribing: ", error);
      setStatus("Subscription failed. Try again.");
      setIsError(true);
    }
  };

  return (
    <div className='w-full py-16 px-4 text-white bg-[#0F172A]'>
      <div className='max-w-[1240px] mx-auto grid lg:grid-cols-3 gap-8'>
        <div className='lg:col-span-2 my-4'>
          <h1 className='md:text-4xl sm:text-3xl text-2xl font-bold py-2 text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-blue-500'>
            Want tips & tricks to optimize your flow?
          </h1>
          <p className='text-gray-400'>
            Sign up for our newsletter and stay up to date.
          </p>
        </div>

        <div className='my-4'>
          <div className='flex flex-col sm:flex-row items-center'>
            <input
              className='p-3 flex w-full rounded-md text-black focus:outline-none'
              type="email"
              placeholder='Enter Email'
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
            <button
              onClick={handleSubscribe}
              className='bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white rounded-md font-medium w-[200px] ml-4 my-6 px-6 py-3 shadow-md transition'
            >
              Notify Me
            </button>
          </div>

          {status && (
            <p className={`text-sm font-semibold ${isError ? 'text-red-400' : 'text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-blue-500'}`}>
              {status}
            </p>
          )}

          <p className='text-gray-400 text-sm mt-2'>
            We care about the protection of your data. Read our{' '}
            <span className='whitespace-nowrap text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-blue-500 font-semibold cursor-pointer'>
              Privacy Policy.
            </span>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Newsletter;
