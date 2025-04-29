import React from 'react';
import {
  FaDribbbleSquare,
  FaFacebookSquare,
  FaGithubSquare,
  FaInstagramSquare,
  FaTwitterSquare,
} from 'react-icons/fa';

export const Footer = () => {
  return (
    <div className='w-full bg-[#0F172A] text-gray-300'>
      <div className='max-w-[1240px] mx-auto py-16 px-4 grid lg:grid-cols-3 gap-8'>
        <div>
          <h1 className='text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-blue-500 cursor-pointer'>
            Data2Dash
          </h1>
          <p className='py-4'>A one click Data Scientist!</p>
          <div className='flex justify-between md:w-[75%] my-6'>
            <FaFacebookSquare size={30} />
            <FaInstagramSquare size={30} />
            <FaTwitterSquare size={30} />
            <FaGithubSquare size={30} />
            {/* <FaDribbbleSquare size={30} /> */}
          </div>
        </div>

        {/* Updated columns */}
        <div className='lg:col-span-2 flex justify-between mt-6'>
          <div>
            <h6 className='font-medium text-gray-400'>Platform</h6>
            <ul>
              <li className='py-2 text-sm'>Dashboard Generator</li>
              <li className='py-2 text-sm'>Visualization Suggestions</li>
              <li className='py-2 text-sm'>LLM Query Interface</li>
              <li className='py-2 text-sm'>Vector Search</li>
            </ul>
          </div>
          <div>
            <h6 className='font-medium text-gray-400'>Resources</h6>
            <ul>
              <li className='py-2 text-sm'>How It Works</li>
              <li className='py-2 text-sm'>Guides & Tutorials</li>
              <li className='py-2 text-sm'>API Reference</li>
              <li className='py-2 text-sm'>Integration Examples</li>
            </ul>
          </div>
          <div>
            <h6 className='font-medium text-gray-400'>Community</h6>
            <ul>
              <li className='py-2 text-sm'>Join Discord / Forum</li>
              <li className='py-2 text-sm'>Blog</li>
              <li className='py-2 text-sm'>Events / Webinars</li>
              <li className='py-2 text-sm'>Contribute on GitHub</li>
            </ul>
          </div>
          {/* <div>
            <h6 className='font-medium text-gray-400'>Company</h6>
            <ul>
              <li className='py-2 text-sm'>About Us</li>
              <li className='py-2 text-sm'>Team</li>
              <li className='py-2 text-sm'>Careers</li>
              <li className='py-2 text-sm'>Contact</li>
            </ul>
          </div> */}
        </div>
      </div>
    </div>
  );
};

export default Footer;
