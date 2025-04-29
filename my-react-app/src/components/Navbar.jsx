import React, { useState } from 'react';
import { AiOutlineClose, AiOutlineMenu } from 'react-icons/ai';


const Navbar = () => {
  const [nav, setNav] = useState(false);

  const handleNav = () => {
    setNav(!nav);
  };

  // const navItems = ['Home', 'Dashboard', 'Guide', 'Pricing','Resources'];
  const navItems = [
    { label: 'Home', href: '#home' },
    { label: 'Dashboard', href: '#dashboard' },
    { label: 'Guide', href: '#guide' },
    {label: 'Demo', href: '#demo'},
    {label: 'Team', href: '#team'},
    { label: 'Pricing', href: '#pricing' },
    { label: 'Resources', href: '#resources' },
  ];
  
  return (
    <div className='fixed top-0 left-0 w-full bg-[#0F172A] shadow-lg z-50'>
      <div className='flex justify-between items-center h-20 max-w-[1240px] mx-auto px-6 text-white'>
        <h1 className='text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-blue-500 cursor-pointer'>
          Data2Dash
        </h1>

        {/* Desktop Nav */}
        <ul className='hidden md:flex space-x-6'>
          {navItems.map((item, idx) => (
            <li key={idx} className='p-2 cursor-pointer group transition duration-300'>
              <a href={item.href} className='text-white group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-purple-500 group-hover:to-blue-500'>
                {item.label}
              </a>
            </li>
          ))}
        </ul>


        {/* Mobile Menu Icon */}
        <div onClick={handleNav} className='block md:hidden cursor-pointer'>
          {nav ? (
            <AiOutlineClose size={25} color="#8B5CF6" />
          ) : (
            <AiOutlineMenu size={25} color="#8B5CF6" />
          )}
        </div>
      </div>

      {/* Mobile Navigation */}
      <ul
        className={`fixed top-0 left-0 w-[60%] h-full bg-[#0F172A] text-white transition-transform duration-500 ${
          nav ? 'translate-x-0' : '-translate-x-full'
        } shadow-md`}
      >
        <h1 className='text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-blue-500 m-6'>
          Data2Dash
        </h1>
        {navItems.map((item, idx) => (
          <li key={idx} className='p-4 border-b border-gray-700 cursor-pointer group transition duration-300'>
            <a href={item.href} onClick={() => setNav(false)} className='text-white group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-purple-500 group-hover:to-blue-500'>
              {item.label}
            </a>
          </li>
        ))}
      </ul>

    </div>
  );
};

export default Navbar;
