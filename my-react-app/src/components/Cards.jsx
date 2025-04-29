// import React from 'react';
// import { useNavigate } from 'react-router-dom';
// import Single from '../assets/single1.png';
// import Double from '../assets/double1.png';
// import Triple from '../assets/triple1.png';

// const Cards = () => {
//   const navigate = useNavigate();

//   const handleStartTrial = (plan) => {
//     navigate(`/checkout/${plan}`);
//   };

//   return (
//     <div className='w-full py-[10rem] px-4' style={{ backgroundColor: '#E7E4EC' }}>
//       <div className='max-w-[1240px] mx-auto grid md:grid-cols-3 gap-8'>

//         {/* Single User */}
//         <div className='group w-full border shadow-xl flex-col py-4 rounded-lg hover:scale-105 hover:bg-[#f8f8f8] duration-300 transition-colors'>
//           <img className='w-20 mx-auto mt-[-3rem] rounded-lg' src={Single} alt="/" />
//           <h2 className='text-2xl font-bold text-center py-8'>Single User</h2>
//           <p className='text-center text-4xl font-bold'>$149</p>
//           <div className='text-center font-medium'>
//             <p className='py-2 border-b mx-8'>500 GB Storage</p>
//             <p className='py-2 border-b mx-8'>1 Granted User</p>
//             <p className='py-2 border-b mx-8'>Send up to 2 GB</p>
//           </div>
//           <button
//             onClick={() => handleStartTrial('single')}
//             className='bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-md font-medium w-[200px] my-6 mx-auto px-6 py-3 shadow-md block'>
//             Start Trial
//           </button>
//         </div>

//         {/* Partnership */}
//         <div className='group w-full border shadow-xl flex-col py-4 rounded-lg hover:scale-105 hover:bg-[#f8f8f8] duration-300 transition-colors'>
//           <img className='w-20 mx-auto mt-[-3rem] rounded-lg' src={Double} alt="/" />
//           <h2 className='text-2xl font-bold text-center py-8'>Partnership</h2>
//           <p className='text-center text-4xl font-bold'>$199</p>
//           <div className='text-center font-medium'>
//             <p className='py-2 border-b mx-8'>1 TB Storage</p>
//             <p className='py-2 border-b mx-8'>3 Users Allowed</p>
//             <p className='py-2 border-b mx-8'>Send up to 10 GB</p>
//           </div>
//           <button
//             onClick={() => handleStartTrial('partnership')}
//             className='bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-md font-medium w-[200px] my-6 mx-auto px-6 py-3 shadow-md block'>
//             Start Trial
//           </button>
//         </div>

//         {/* Group Account */}
//         <div className='group w-full border shadow-xl flex-col py-4 rounded-lg hover:scale-105 hover:bg-[#f8f8f8] duration-300 transition-colors'>
//           <img className='w-20 mx-auto mt-[-3rem] rounded-lg' src={Triple} alt="/" />
//           <h2 className='text-2xl font-bold text-center py-8'>Group Account</h2>
//           <p className='text-center text-4xl font-bold'>$299</p>
//           <div className='text-center font-medium'>
//             <p className='py-2 border-b mx-8'>5 TB Storage</p>
//             <p className='py-2 border-b mx-8'>10 Users Allowed</p>
//             <p className='py-2 border-b mx-8'>Send up to 20 GB</p>
//           </div>
//           <button
//             onClick={() => handleStartTrial('group')}
//             className='bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-md font-medium w-[200px] my-6 mx-auto px-6 py-3 shadow-md block'>
//             Start Trial
//           </button>
//         </div>

//       </div>
//     </div>
//   );
// };

// export default Cards;

// import React from 'react';
// import { useNavigate } from 'react-router-dom';
// import Single from '../assets/single1.png';
// import Double from '../assets/double1.png';
// import Triple from '../assets/triple1.png';
// import Quad from '../assets/quad1.png';  // New image for the fourth card

// const Cards = () => {
//   const navigate = useNavigate();

//   const handleStartTrial = (plan) => {
//     navigate(`/checkout/${plan}`);
//   };

//   return (
//     <div className='w-full py-[10rem] px-4' style={{ backgroundColor: '#E7E4EC' }}>
//       <div className='max-w-[1240px] mx-auto grid md:grid-cols-4 gap-8'>  {/* Updated grid to 4 columns */}

//         {/* Single User */}
//         <div className='group w-full border shadow-xl flex-col py-4 rounded-lg hover:scale-105 hover:bg-[#f8f8f8] duration-300 transition-colors'>
//           <img className='w-20 mx-auto mt-[-3rem] rounded-lg' src={Single} alt="/" />
//           <h2 className='text-2xl font-bold text-center py-8'>Tier 1</h2>
//           <p className='text-center text-4xl font-bold'>Free</p>
//           <div className='text-center font-medium'>
//             <p className='py-2 border-b mx-8'>1 dashboard</p>
//             <p className='py-2 border-b mx-8'>1 chat</p>
//             <p className='py-2 border-b mx-8'>2 Q/A retrievals/day</p>
//           </div>
//           <button
//             onClick={() => handleStartTrial('single')}
//             className='bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-md font-medium w-[200px] my-6 mx-auto px-6 py-3 shadow-md block'>
//             Start Trial
//           </button>
//         </div>

//         {/* Partnership */}
//         <div className='group w-full border shadow-xl flex-col py-4 rounded-lg hover:scale-105 hover:bg-[#f8f8f8] duration-300 transition-colors'>
//           <img className='w-20 mx-auto mt-[-3rem] rounded-lg' src={Double} alt="/" />
//           <h2 className='text-2xl font-bold text-center py-8'>Tier 2</h2>
//           <p className='text-center text-4xl font-bold'>$20</p>
//           <div className='text-center font-medium'>
//             <p className='py-2 border-b mx-8'>5 dashboards</p>
//             <p className='py-2 border-b mx-8'>5 chats</p>
//             <p className='py-2 border-b mx-8'>50 Q/A retrievals/day</p>
//           </div>
//           <button
//             onClick={() => handleStartTrial('partnership')}
//             className='bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-md font-medium w-[200px] my-6 mx-auto px-6 py-3 shadow-md block'>
//             Start Trial
//           </button>
//         </div>

//         {/* Group Account */}
//         <div className='group w-full border shadow-xl flex-col py-4 rounded-lg hover:scale-105 hover:bg-[#f8f8f8] duration-300 transition-colors'>
//           <img className='w-20 mx-auto mt-[-3rem] rounded-lg' src={Triple} alt="/" />
//           <h2 className='text-2xl font-bold text-center py-8'>Tier 3</h2>
//           <p className='text-center text-4xl font-bold'>$30</p>
//           <div className='text-center font-medium'>
//             <p className='py-2 border-b mx-8'>20 dashboards</p>
//             <p className='py-2 border-b mx-8'>unlimited chats</p>
//             <p className='py-2 border-b mx-8'>unlimited Q/A retrievals/day</p>
//             {/* <p className='py-2 border-b mx-8'>multi-agent support</p> */}

//           </div>
//           <button
//             onClick={() => handleStartTrial('group')}
//             className='bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-md font-medium w-[200px] my-6 mx-auto px-6 py-3 shadow-md block'>
//             Start Trial
//           </button>
//         </div>

//         {/* New Card - Quadruple Plan */}
//         <div className='group w-full border shadow-xl flex-col py-4 rounded-lg hover:scale-105 hover:bg-[#f8f8f8] duration-300 transition-colors'>
//           <img className='w-20 mx-auto mt-[-3rem] rounded-lg' src={Quad} alt="/" />  {/* New image for Quad Plan */}
//           <h2 className='text-2xl font-bold text-center py-8'>Tier 4</h2>
//           <p className='text-center text-4xl font-bold'>$100</p>
//           <div className='text-center font-medium'>
//             <p className='py-2 border-b mx-8'>enterprise integration</p>
//             <p className='py-2 border-b mx-8'>multi-agent support</p>
//             <p className='py-2 border-b mx-8'>customized plan</p>
//           </div>
//           <button
//             onClick={() => handleStartTrial('quad')}
//             className='bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-md font-medium w-[200px] my-6 mx-auto px-6 py-3 shadow-md block'>
//             Start Trial
//           </button>
//         </div>

//       </div>
//     </div>
//   );
// };

// export default Cards;

import React from 'react';
import { useNavigate } from 'react-router-dom';

// Card images (correct paths)
import Single from '../assets/single1.png';  // Corrected path
import Double from '../assets/double1.png';  // Corrected path
import Triple from '../assets/triple1.png';  // Corrected path
import Quad from '../assets/quad1.png';      // Corrected path

const Cards = () => {
  const navigate = useNavigate();

  const handleStartTrial = (plan) => {
    navigate(`/checkout/${plan}`);
  };

  return (
    <div className="w-full py-[10rem] px-4" style={{ backgroundColor: '#E7E4EC' }}>
      <div className="max-w-[1240px] mx-auto grid md:grid-cols-4 gap-8">
        {/* Single User */}
        <div className="group w-full border shadow-xl flex-col py-4 rounded-lg hover:scale-105 hover:bg-[#f8f8f8] duration-300 transition-colors">
          <img className="w-20 mx-auto mt-[-3rem] rounded-lg" src={Single} alt="Single" />
          <h2 className="text-2xl font-bold text-center py-8">Tier 1</h2>
          <p className="text-center text-4xl font-bold">Free</p>
          <div className="text-center font-medium">
            <p className="py-2 border-b mx-8">1 dashboard</p>
            <p className="py-2 border-b mx-8">1 chat</p>
            <p className="py-2 border-b mx-8">2 Q/A retrievals/day</p>
          </div>
          <button
            onClick={() => handleStartTrial('free')}
            className="bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-md font-medium w-[200px] my-6 mx-auto px-6 py-3 shadow-md block"
          >
            Start Trial
          </button>
        </div>

        {/* Tier 2 */}
        <div className="group w-full border shadow-xl flex-col py-4 rounded-lg hover:scale-105 hover:bg-[#f8f8f8] duration-300 transition-colors">
          <img className="w-20 mx-auto mt-[-3rem] rounded-lg" src={Double} alt="Partnership" />
          <h2 className="text-2xl font-bold text-center py-8">Tier 2</h2>
          <p className="text-center text-4xl font-bold">$20</p>
          <div className="text-center font-medium">
            <p className="py-2 border-b mx-8">5 dashboards</p>
            <p className="py-2 border-b mx-8">5 chats</p>
            <p className="py-2 border-b mx-8">50 Q/A retrievals/day</p>
          </div>
          <button
            onClick={() => handleStartTrial('tier2')}
            className="bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-md font-medium w-[200px] my-6 mx-auto px-6 py-3 shadow-md block"
          >
            Start Trial
          </button>
        </div>

        {/* Tier 3 */}
        <div className="group w-full border shadow-xl flex-col py-4 rounded-lg hover:scale-105 hover:bg-[#f8f8f8] duration-300 transition-colors">
          <img className="w-20 mx-auto mt-[-3rem] rounded-lg" src={Triple} alt="Group Account" />
          <h2 className="text-2xl font-bold text-center py-8">Tier 3</h2>
          <p className="text-center text-4xl font-bold">$30</p>
          <div className="text-center font-medium">
            <p className="py-2 border-b mx-8">20 dashboards</p>
            <p className="py-2 border-b mx-8">unlimited chats</p>
            <p className="py-2 border-b mx-8">unlimited Q/A retrievals/day</p>
          </div>
          <button
            onClick={() => handleStartTrial('tier3')}
            className="bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-md font-medium w-[200px] my-6 mx-auto px-6 py-3 shadow-md block"
          >
            Start Trial
          </button>
        </div>

        {/* Tier 4 */}
        <div className="group w-full border shadow-xl flex-col py-4 rounded-lg hover:scale-105 hover:bg-[#f8f8f8] duration-300 transition-colors">
          <img className="w-20 mx-auto mt-[-3rem] rounded-lg" src={Quad} alt="Enterprise" />
          <h2 className="text-2xl font-bold text-center py-8">Tier 4</h2>
          <p className="text-center text-4xl font-bold">$100</p>
          <div className="text-center font-medium">
            <p className="py-2 border-b mx-8">enterprise integration</p>
            <p className="py-2 border-b mx-8">multi-agent support</p>
            <p className="py-2 border-b mx-8">customized plan</p>
          </div>
          <button
            onClick={() => handleStartTrial('tier4')}
            className="bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-md font-medium w-[200px] my-6 mx-auto px-6 py-3 shadow-md block"
          >
            Start Trial
          </button>
        </div>
      </div>
    </div>
  );
};

export default Cards;
