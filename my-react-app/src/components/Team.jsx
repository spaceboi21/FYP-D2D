import React from 'react';
import abbasImage from '../assets/abbas.jpg';
import duaImage from '../assets/dua.jpg';
import hamzaImage from '../assets/hamz.jpg';

const Team = () => {
  const teamMembers = [
    {
      name: "Muhammad Abbas ",
      email: "ma_abbas2001@hotmail.com",
      imageUrl: abbasImage,
    },
    {
      name: "Dua Khan",
      email: "kduaa03@gmail.com",
      imageUrl: duaImage,
    },
    {
      name: "Muhammad Hamza",
      email: "muhamza378@gmail.com",
      imageUrl: hamzaImage,
    },
  ];

  return (
    <div className='w-full py-16 px-4 text-white bg-[#0F172A]'>
      <div className='max-w-[1240px] mx-auto'>
        <h1 className='md:text-4xl sm:text-3xl text-2xl font-bold py-2 text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-blue-500 text-center mb-8'>
          Our Team
        </h1>
        <div className='grid md:grid-cols-3 gap-8'>
          {teamMembers.map((member, index) => (
            <div key={index} className='text-center p-6 rounded-lg bg-[#1E293B] hover:bg-[#2D3748] transition duration-300'>
              <img
                src={member.imageUrl}
                alt={member.name}
                className='w-32 h-32 rounded-full mx-auto mb-4 object-cover'
              />
              <h3 className='text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-500 to-blue-500'>
                {member.name}
              </h3>
              <p className='text-gray-400 mt-2'>{member.email}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Team;
