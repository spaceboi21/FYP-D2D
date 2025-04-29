import React from 'react'
import Laptop from '../assets/laptop2.png'

export const Analytics = () => {
  return (
    <div className='w-full py-16 px-4' style={{ backgroundColor: '#E7E4EC' }}>
        <div className='max-w-[1240px] mx-auto grid md:grid-cols-2'>
            <img
                src={Laptop}
                alt="/"
                style={{
                    width: '80%',
                    maxWidth: '700px',
                    borderRadius: '8px',
                    margin: 'auto',
                    display: 'block'
                }}
                className='mx-auto my-4'
            />
            <div className='flex flex-col justify-center'>
                <p className='text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-blue-500 font-bold'>
                    DATA ANALYTICS DASHBOARD
                </p>
                <h1 className='md:text-4xl sm:text-3xl text-2xl font-bold py-2 text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-blue-500'>
                    Centralize Your Data. Maximize Insight.
                </h1>
                <p className='text-gray-700 font-medium'>
                    Drive smarter decisions with real-time analytics. Unify your data, accelerate performance, and innovate faster than ever before.
                </p>
                <button className='w-[200px] rounded-md font-medium my-6 mx-auto md:mx-0 py-3 text-white transition 
                    bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 shadow-md'>

                <a 
                            href="http://127.0.0.1:5001/" 
                            className="block w-full h-full text-center"
                            target="_blank" 
                            rel="noopener noreferrer"
                        >
                    Get Started
                    </a>
                </button>
            </div>
        </div>
    </div>
  );
};

export default Analytics;
