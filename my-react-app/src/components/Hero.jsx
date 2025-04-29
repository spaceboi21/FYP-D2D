import React from 'react';
import ReactTyped from "react-typed";

const Hero = () => {
  return (
    <div className='text-white bg-[#0F172A] min-h-screen flex items-center justify-center px-4'>
      <div className='max-w-[900px] w-full py-16 rounded-lg text-center'>
        {/* Main Heading */}
        <h1 className='text-4xl sm:text-5xl md:text-7xl lg:text-8xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-blue-500 leading-tight'>
          Grow with data.
        </h1>

        {/* Typed Line */}
        <div className='flex flex-wrap justify-center items-center text-center pt-6'>
          <p className='text-xl sm:text-2xl md:text-4xl font-semibold text-gray-300'>
            Unlock to drive
          </p>
          <ReactTyped
            className='text-xl sm:text-2xl md:text-4xl font-bold pl-2 text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-blue-500'
            strings={['Growth', 'Efficiency', 'Innovation', 'Success', 'Insights']}
            typeSpeed={120}
            backSpeed={140}
            loop
          />
        </div>

        {/* Subtitle */}
        <p className='text-base sm:text-lg md:text-xl font-medium text-gray-400 mt-4 max-w-xl mx-auto'>
          Make smarter, faster, and data-driven decisions effortlessly.
        </p>

        {/* Button */}
        <button 
          className='w-[200px] rounded-md font-semibold my-6 mx-auto py-3 text-white transition 
                     bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600'
        >
          <a 
            href="http://127.0.0.1:5001/" 
            className="block w-full h-full text-center"
            target="_blank" 
            rel="noopener noreferrer"
          >
            Try Now
          </a>
        </button>

      </div>
    </div>
  );
};

export default Hero;
